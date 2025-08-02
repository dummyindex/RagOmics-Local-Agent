#!/usr/bin/env python3
"""Comprehensive test to verify all outputs are saved correctly."""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

# Try imports with fallback
try:
    from models import (
        AnalysisTree, AnalysisNode, NodeState, GenerationMode,
        NewFunctionBlock, FunctionBlockType, StaticConfig, Arg
    )
    from analysis_tree_management import AnalysisTreeManager
    from analysis_tree_management.node_executor import NodeExecutor
    from job_executors import ExecutorManager
except ImportError:
    from ragomics_agent_local.models import (
        AnalysisTree, AnalysisNode, NodeState, GenerationMode,
        NewFunctionBlock, FunctionBlockType, StaticConfig, Arg
    )
    from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
    from ragomics_agent_local.analysis_tree_management.node_executor import NodeExecutor
    from ragomics_agent_local.job_executors import ExecutorManager


def create_test_function_block():
    """Create a test function block that generates various outputs."""
    # This code follows the standard PythonExecutor wrapper format
    code = '''
def run(adata, **parameters):
    """Test function block that generates multiple outputs."""
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    from datetime import datetime
    
    print("Starting test function block execution...")
    print(f"Input adata shape: {adata.shape}")
    print(f"Parameters: {parameters}")
    
    # Generate some additional test data
    data = np.random.randn(100, 50)
    
    # Create new AnnData from random data for testing
    test_adata = sc.AnnData(data)
    test_adata.obs_names = [f"cell_{i}" for i in range(data.shape[0])]
    test_adata.var_names = [f"gene_{i}" for i in range(data.shape[1])]
    
    # Create figures directory
    import os
    os.makedirs("/workspace/output/figures", exist_ok=True)
    
    # Generate figure 1: Histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(data.flatten(), bins=50, alpha=0.7)
    ax.set_title("Test Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.savefig("/workspace/output/figures/histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure: histogram.png")
    
    # Generate figure 2: Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(data[:20, :20], cmap='viridis', aspect='auto')
    ax.set_title("Test Heatmap")
    plt.colorbar(im, ax=ax)
    plt.savefig("/workspace/output/figures/heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure: heatmap.png")
    
    # Generate figure 3: Line plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i in range(5):
        ax.plot(data[:, i], label=f"Series {i+1}", alpha=0.7)
    ax.set_title("Test Line Plot")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    plt.savefig("/workspace/output/figures/lineplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure: lineplot.png")
    
    # Write to stderr for testing
    import sys
    print("This is a warning message", file=sys.stderr)
    print("Another stderr message for testing", file=sys.stderr)
    
    # Create metadata
    metadata = {
        "test_run": True,
        "timestamp": str(datetime.now()),
        "data_shape": list(test_adata.shape),
        "figures_generated": ["histogram.png", "heatmap.png", "lineplot.png"],
        "parameters_used": parameters
    }
    
    # Save metadata
    with open("/workspace/output/analysis_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to: analysis_metadata.json")
    
    # Save additional CSV output
    df = pd.DataFrame(data[:10, :10])
    df.to_csv("/workspace/output/test_data.csv", index=False)
    print("Saved CSV data")
    
    print("Test function block completed successfully!")
    
    # Return the test AnnData
    return test_adata
'''
    
    # Create function block
    static_config = StaticConfig(
        args=[
            Arg(name="test_param", value_type="str", description="Test parameter", 
                optional=True, default_value="default")
        ],
        description="Test function block for output verification",
        tag="test"
    )
    
    block = NewFunctionBlock(
        name="test_output_block",
        type=FunctionBlockType.PYTHON,
        description="Test block that generates multiple outputs",
        static_config=static_config,
        code=code,
        requirements="pandas\nnumpy\nmatplotlib\nscanpy",
        parameters={"test_param": "test_value"}
    )
    
    return block


def verify_job_outputs(job_dir: Path) -> Dict:
    """Verify all expected outputs in a job directory."""
    results = {
        "job_dir": str(job_dir),
        "exists": job_dir.exists(),
        "output_dir": False,
        "figures_dir": False,
        "logs_dir": False,
        "stdout": False,
        "stderr": False,
        "figures": [],
        "output_files": [],
        "metadata": False,
        "execution_summary": False
    }
    
    if not job_dir.exists():
        return results
    
    # Check directories
    output_dir = job_dir / "output"
    figures_dir = job_dir / "figures"
    logs_dir = job_dir / "logs"
    
    results["output_dir"] = output_dir.exists()
    results["figures_dir"] = figures_dir.exists()
    results["logs_dir"] = logs_dir.exists()
    
    # Check logs
    if logs_dir.exists():
        stdout_file = logs_dir / "stdout.txt"
        stderr_file = logs_dir / "stderr.txt"
        results["stdout"] = stdout_file.exists()
        results["stderr"] = stderr_file.exists()
        
        if stdout_file.exists():
            with open(stdout_file) as f:
                content = f.read()
                results["stdout_content_size"] = len(content)
                results["stdout_has_content"] = len(content) > 0
        
        if stderr_file.exists():
            with open(stderr_file) as f:
                content = f.read()
                results["stderr_content_size"] = len(content)
                results["stderr_has_content"] = len(content) > 0
    
    # Check figures
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        results["figures"] = [f.name for f in figures]
        results["figures_count"] = len(figures)
    
    # Check output files
    if output_dir.exists():
        output_files = []
        for f in output_dir.glob("*"):
            if f.is_file():
                output_files.append({
                    "name": f.name,
                    "size": f.stat().st_size
                })
        results["output_files"] = output_files
        results["output_files_count"] = len(output_files)
        
        # Check for specific files
        metadata_file = output_dir / "analysis_metadata.json"
        results["metadata"] = metadata_file.exists()
        
        anndata_file = output_dir / "anndata.h5ad"
        results["anndata"] = anndata_file.exists()
    
    # Check execution summary
    summary_file = job_dir / "execution_summary.json"
    results["execution_summary"] = summary_file.exists()
    if summary_file.exists():
        with open(summary_file) as f:
            results["execution_summary_content"] = json.load(f)
    
    return results


def verify_node_outputs(node_dir: Path) -> Dict:
    """Verify node-level outputs."""
    results = {
        "node_dir": str(node_dir),
        "exists": node_dir.exists(),
        "output_dir": False,
        "figures_dir": False,
        "logs_dir": False,
        "latest_job_link": False,
        "job_dirs": [],
        "node_figures": [],
        "node_output_files": []
    }
    
    if not node_dir.exists():
        return results
    
    # Check node-level directories
    output_dir = node_dir / "output"
    figures_dir = node_dir / "figures"
    logs_dir = node_dir / "logs"
    
    results["output_dir"] = output_dir.exists()
    results["figures_dir"] = figures_dir.exists()
    results["logs_dir"] = logs_dir.exists()
    
    # Check latest job link
    latest_link = node_dir / "latest_job"
    results["latest_job_link"] = latest_link.exists()
    if latest_link.exists() and latest_link.is_symlink():
        results["latest_job_target"] = str(latest_link.resolve().name)
    
    # List job directories
    job_dirs = []
    for item in node_dir.iterdir():
        if item.is_dir() and item.name.startswith("job_"):
            job_dirs.append(item.name)
    results["job_dirs"] = sorted(job_dirs)
    results["job_count"] = len(job_dirs)
    
    # Check node-level figures
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        results["node_figures"] = [f.name for f in figures]
    results["node_figures_count"] = len(results.get("node_figures", []))
    
    # Check node-level output files
    if output_dir.exists():
        output_files = []
        for f in output_dir.glob("*"):
            if f.is_file():
                output_files.append(f.name)
        results["node_output_files"] = output_files
    results["node_output_files_count"] = len(results.get("node_output_files", []))
    
    return results


def test_single_node_execution():
    """Test execution of a single node with output verification."""
    print("="*60)
    print("Test: Single Node Execution with Output Verification")
    print("="*60)
    
    # Setup
    output_base = Path("test_outputs/output_verification")
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create test data
    print("\n1. Creating test data...")
    import numpy as np
    test_data = np.random.randn(50, 20)
    
    # Save as h5ad for PythonExecutor compatibility
    try:
        import scanpy as sc
        adata = sc.AnnData(test_data)
        adata.obs_names = [f"cell_{i}" for i in range(test_data.shape[0])]
        adata.var_names = [f"gene_{i}" for i in range(test_data.shape[1])]
        input_file = output_base / "input_data.h5ad"
        adata.write(input_file)
        print(f"   Created input: {input_file}")
    except ImportError:
        # Fallback to CSV if scanpy not available
        import pandas as pd
        df = pd.DataFrame(test_data)
        input_file = output_base / "input_data.csv"
        df.to_csv(input_file, index=False)
        print(f"   Created input: {input_file} (CSV fallback)")
    
    # Create tree and node
    print("\n2. Setting up analysis tree...")
    tree_manager = AnalysisTreeManager()
    tree = tree_manager.create_tree(
        user_request="Test output verification",
        input_data_path=str(input_file),
        max_nodes=5
    )
    
    # Create test function block
    test_block = create_test_function_block()
    
    # Add as root node
    node = tree_manager.add_root_node(test_block)
    print(f"   Created node: {node.id[:8]}...")
    
    # Create executor
    print("\n3. Executing node...")
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Execute node
    state, output_path = node_executor.execute_node(
        node=node,
        tree=tree,
        input_path=input_file,
        output_base_dir=output_base
    )
    
    print(f"   Execution state: {state}")
    print(f"   Output path: {output_path}")
    
    # Verify outputs
    print("\n4. Verifying outputs...")
    
    # Find the node directory
    node_dir = output_base / tree.id / node.id
    print(f"\n   Node directory: {node_dir}")
    
    # Verify node-level outputs
    node_results = verify_node_outputs(node_dir)
    
    print("\n   Node-level verification:")
    print(f"     - Node directory exists: {node_results['exists']}")
    print(f"     - Output directory: {node_results['output_dir']}")
    print(f"     - Figures directory: {node_results['figures_dir']}")
    print(f"     - Logs directory: {node_results['logs_dir']}")
    print(f"     - Latest job link: {node_results['latest_job_link']}")
    print(f"     - Number of jobs: {node_results['job_count']}")
    print(f"     - Node figures: {node_results['node_figures_count']} files")
    print(f"     - Node outputs: {node_results['node_output_files_count']} files")
    
    # Verify job-level outputs for each job
    for job_name in node_results['job_dirs']:
        job_dir = node_dir / job_name
        job_results = verify_job_outputs(job_dir)
        
        print(f"\n   Job-level verification ({job_name}):")
        print(f"     - Job directory exists: {job_results['exists']}")
        print(f"     - Output directory: {job_results['output_dir']}")
        print(f"     - Figures directory: {job_results['figures_dir']}")
        print(f"     - Logs directory: {job_results['logs_dir']}")
        print(f"     - stdout.txt: {job_results['stdout']}")
        print(f"     - stderr.txt: {job_results['stderr']}")
        print(f"     - Figures: {job_results['figures_count']} files")
        if job_results['figures']:
            for fig in job_results['figures']:
                print(f"       * {fig}")
        print(f"     - Output files: {job_results['output_files_count']} files")
        if job_results['output_files']:
            for f in job_results['output_files']:
                print(f"       * {f['name']} ({f['size']} bytes)")
        print(f"     - Metadata file: {job_results['metadata']}")
        print(f"     - Execution summary: {job_results['execution_summary']}")
    
    # Check redundancy (files in both job and node level)
    if node_results['job_dirs']:
        latest_job = node_results['job_dirs'][-1]
        job_dir = node_dir / latest_job
        job_results = verify_job_outputs(job_dir)
        
        print("\n5. Checking redundancy (files in both levels):")
        
        # Compare figures
        job_figures = set(job_results.get('figures', []))
        node_figures = set(node_results.get('node_figures', []))
        common_figures = job_figures & node_figures
        
        print(f"   Figures:")
        print(f"     - In job only: {job_figures - node_figures}")
        print(f"     - In node only: {node_figures - job_figures}")
        print(f"     - In both (redundant): {common_figures}")
        
        # Check if files are identical
        if common_figures and job_results['figures_dir'] and node_results['figures_dir']:
            for fig in list(common_figures)[:2]:  # Check first 2
                job_fig = job_dir / "figures" / fig
                node_fig = node_dir / "figures" / fig
                if job_fig.exists() and node_fig.exists():
                    same_size = job_fig.stat().st_size == node_fig.stat().st_size
                    print(f"     - {fig}: Same size: {same_size}")
    
    # Summary
    print("\n6. Summary:")
    success_criteria = [
        ("Node executed successfully", state == NodeState.COMPLETED),
        ("Node directory created", node_results['exists']),
        ("Job directories created", node_results['job_count'] > 0),
        ("Stdout captured", any(verify_job_outputs(node_dir / j).get('stdout', False) 
                              for j in node_results['job_dirs'])),
        ("Stderr captured", any(verify_job_outputs(node_dir / j).get('stderr', False) 
                              for j in node_results['job_dirs'])),
        ("Figures generated", node_results['node_figures_count'] > 0),
        ("Output files created", node_results['node_output_files_count'] > 0),
        ("Redundancy maintained", len(common_figures) > 0 if 'common_figures' in locals() else False)
    ]
    
    for criterion, passed in success_criteria:
        status = "✓" if passed else "✗"
        print(f"   {status} {criterion}")
    
    all_passed = all(passed for _, passed in success_criteria)
    print(f"\n   Overall: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    
    return all_passed


def test_multi_node_pipeline():
    """Test multi-node pipeline with output verification."""
    print("\n" + "="*60)
    print("Test: Multi-Node Pipeline with Output Verification")
    print("="*60)
    
    # Setup
    output_base = Path("test_outputs/output_verification_pipeline")
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create initial data
    print("\n1. Creating initial data...")
    import numpy as np
    
    data = np.random.randn(100, 50)
    
    # Save as h5ad for PythonExecutor compatibility
    try:
        import scanpy as sc
        adata = sc.AnnData(data)
        adata.obs_names = [f"cell_{i}" for i in range(data.shape[0])]
        adata.var_names = [f"gene_{i}" for i in range(data.shape[1])]
        input_file = output_base / "initial_input.h5ad"
        adata.write(input_file)
        print(f"   Created input: {input_file}")
    except ImportError:
        # Fallback to CSV
        import pandas as pd
        df = pd.DataFrame(data)
        input_file = output_base / "initial_input.csv"
        df.to_csv(input_file, index=False)
        print(f"   Created input: {input_file} (CSV fallback)")
    
    # Create tree
    print("\n2. Setting up pipeline...")
    tree_manager = AnalysisTreeManager()
    tree = tree_manager.create_tree(
        user_request="Test multi-node pipeline",
        input_data_path=str(input_file),
        max_nodes=10
    )
    
    # Create multiple nodes
    test_blocks = []
    for i in range(3):
        block = create_test_function_block()
        block.name = f"test_block_{i+1}"
        block.parameters = {"test_param": f"node_{i+1}"}
        test_blocks.append(block)
    
    # Build pipeline
    root_node = tree_manager.add_root_node(test_blocks[0])
    child_nodes = tree_manager.add_child_nodes(root_node.id, test_blocks[1:])
    
    print(f"   Created pipeline with {len(tree.nodes)} nodes")
    
    # Execute pipeline
    print("\n3. Executing pipeline...")
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    results = []
    
    # Execute root
    print(f"\n   Executing root node...")
    state, output = node_executor.execute_node(
        node=root_node,
        tree=tree,
        input_path=input_file,
        output_base_dir=output_base
    )
    results.append((root_node.id, state))
    print(f"     State: {state}")
    
    # Execute children
    for child in child_nodes:
        print(f"\n   Executing child node {child.id[:8]}...")
        state, output = node_executor.execute_node(
            node=child,
            tree=tree,
            input_path=root_node.output_data_id if root_node.output_data_id else input_file,
            output_base_dir=output_base
        )
        results.append((child.id, state))
        print(f"     State: {state}")
    
    # Verify all outputs
    print("\n4. Verifying all node outputs...")
    
    for node_id, exec_state in results:
        node_dir = output_base / tree.id / node_id
        node_results = verify_node_outputs(node_dir)
        
        print(f"\n   Node {node_id[:8]}:")
        print(f"     - Execution state: {exec_state}")
        print(f"     - Jobs created: {node_results['job_count']}")
        print(f"     - Figures: {node_results['node_figures_count']}")
        print(f"     - Outputs: {node_results['node_output_files_count']}")
        
        # Check latest job
        if node_results['job_dirs']:
            latest_job = node_results['job_dirs'][-1]
            job_dir = node_dir / latest_job
            job_results = verify_job_outputs(job_dir)
            
            print(f"     - Latest job has stdout: {job_results['stdout']}")
            print(f"     - Latest job has stderr: {job_results['stderr']}")
    
    # Summary
    print("\n5. Pipeline Summary:")
    all_completed = all(state == NodeState.COMPLETED for _, state in results)
    print(f"   Total nodes: {len(results)}")
    print(f"   Completed: {sum(1 for _, s in results if s == NodeState.COMPLETED)}")
    print(f"   Failed: {sum(1 for _, s in results if s == NodeState.FAILED)}")
    print(f"   Overall: {'✓ PASSED' if all_completed else '✗ FAILED'}")
    
    return all_completed


def main():
    """Run all output verification tests."""
    print("="*60)
    print("Output Verification Test Suite")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    
    results = []
    
    # Test 1: Single node
    try:
        success = test_single_node_execution()
        results.append(("Single Node Execution", success))
    except Exception as e:
        print(f"\nError in single node test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Single Node Execution", False))
    
    # Test 2: Multi-node pipeline
    try:
        success = test_multi_node_pipeline()
        results.append(("Multi-Node Pipeline", success))
    except Exception as e:
        print(f"\nError in pipeline test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multi-Node Pipeline", False))
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} passed")
    
    # List output directories for manual inspection
    print("\n" + "="*60)
    print("Output Directories for Manual Inspection:")
    print("="*60)
    
    for output_dir in ["test_outputs/output_verification", "test_outputs/output_verification_pipeline"]:
        path = Path(output_dir)
        if path.exists():
            print(f"\n{output_dir}:")
            # Show tree structure (first 2 levels)
            for item in path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(path)
                    depth = len(rel_path.parts)
                    if depth <= 4:  # Limit depth
                        indent = "  " * (depth - 1)
                        size = item.stat().st_size
                        print(f"{indent}- {rel_path.name} ({size} bytes)")
    
    return all(passed for _, passed in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)