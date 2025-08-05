#!/usr/bin/env python3
"""Clustering benchmark test with both mock and real LLM."""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_utils import get_test_output_dir

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
from ragomics_agent_local.models import NodeState
from ragomics_agent_local.tests.test_main_agent_mocked import (
    create_clustering_pipeline,
    create_predefined_function_blocks,
    MockOrchestratorAgent,
    MockFunctionCreator,
    MockFunctionSelector
)
from unittest.mock import patch
import scanpy as sc
import numpy as np


def create_zebrafish_test_data(output_dir: Path) -> Path:
    """Create zebrafish test data."""
    test_data_dir = output_dir / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_data_dir / "zebrafish.h5ad"
    
    # Create realistic zebrafish-like test data
    n_obs, n_vars = 500, 200  # Smaller for testing
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    adata = sc.AnnData(X=X.astype(np.float32))
    adata.obs_names = [f'Cell_{i:04d}' for i in range(n_obs)]
    adata.var_names = [f'Gene_{i:04d}' for i in range(n_vars)]
    
    # Add cell type annotations (ground truth for clustering)
    cell_types = ['Neural', 'Muscle', 'Epithelial', 'Blood', 'Germ']
    adata.obs['cell_type'] = np.random.choice(cell_types, size=n_obs)
    
    # Add some metadata
    adata.obs['n_counts'] = np.random.poisson(1000, n_obs)
    adata.obs['n_genes'] = np.random.poisson(200, n_obs)
    
    adata.write(test_file)
    return test_file


def run_clustering_benchmark_mock():
    """Run clustering benchmark with mock LLM."""
    print("\n" + "="*80)
    print("CLUSTERING BENCHMARK - MOCK LLM")
    print("="*80)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_test_output_dir() / f"clustering_mock_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output Directory: {output_dir}")
    
    # Create test data
    test_file = create_zebrafish_test_data(output_dir)
    print(f"📊 Test Data: {test_file}")
    print(f"   Size: {test_file.stat().st_size / 1024:.1f} KB")
    
    # Setup mock agents
    pipeline = create_clustering_pipeline()
    predefined_blocks = create_predefined_function_blocks()
    
    mock_orchestrator = MockOrchestratorAgent(pipeline)
    mock_creator = MockFunctionCreator(predefined_blocks)
    mock_selector = MockFunctionSelector()
    
    # Create main agent with mocks
    main_agent = MainAgent()
    main_agent.orchestrator = mock_orchestrator
    main_agent.function_creator = mock_creator
    main_agent.function_selector = mock_selector
    
    # Mock node executor to avoid actual execution
    with patch.object(main_agent.node_executor, 'execute_node') as mock_execute:
        def mock_execution(node, tree, input_path, output_base_dir):
            # Create proper output structure following docs
            from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
            tree_manager = AnalysisTreeManager(tree)
            
            # Get nodes directory - corrected structure (no tree_ prefix)
            tree_dir = output_base_dir / tree.id
            nodes_dir = tree_dir / "nodes"
            nodes_dir.mkdir(parents=True, exist_ok=True)
            
            # Create node directory
            node_paths = tree_manager.create_node_directory(node.id, nodes_dir)
            
            # Create job directory
            job_dir = tree_manager.create_job_directory(node.id, node_paths['node_dir'])
            
            # Simulate execution
            output_file = job_dir / "output" / "_node_anndata.h5ad"
            if Path(input_path).exists():
                import shutil
                if Path(input_path).is_file():
                    shutil.copy2(input_path, output_file)
                else:
                    # Handle directory input
                    for f in Path(input_path).glob("*.h5ad"):
                        shutil.copy2(f, output_file)
                        break
            else:
                output_file.touch()
            
            # Copy outputs to node
            tree_manager.copy_outputs_to_node(job_dir, node_paths['node_dir'])
            
            return (NodeState.COMPLETED, str(node_paths['node_dir'] / "outputs"))
        
        mock_execute.side_effect = mock_execution
        
        # Run analysis
        print("\n🚀 Running Mock Analysis...")
        user_request = """
        Your job is to benchmark different clustering methods on the given dataset.
        Apply quality control, normalization, PCA, and then test multiple clustering
        methods (Leiden at different resolutions). Calculate clustering metrics.
        """
        
        start_time = datetime.now()
        result = main_agent.run_analysis(
            input_data_path=test_file,
            user_request=user_request,
            output_dir=output_dir,
            max_nodes=5,
            max_children=1,
            verbose=True
        )
        end_time = datetime.now()
        
        print("\n" + "-"*80)
        print("📈 MOCK RESULTS")
        print("-"*80)
        print(f"✅ Total Nodes: {result['total_nodes']}")
        print(f"✅ Completed: {result['completed_nodes']}")
        print(f"❌ Failed: {result['failed_nodes']}")
        print(f"⏱️ Execution Time: {(end_time - start_time).total_seconds():.2f} seconds")
        
        # Load and display tree structure
        tree_file = Path(result['tree_file'])
        if tree_file.exists():
            with open(tree_file) as f:
                tree_data = json.load(f)
            
            print("\n🌲 Analysis Tree:")
            print("-"*40)
            nodes = tree_data.get('nodes', {})
            sorted_nodes = sorted(nodes.values(), key=lambda x: x.get('level', 0))
            
            for node in sorted_nodes:
                indent = "  " * node.get('level', 0)
                name = node['function_block']['name']
                state = node.get('state', 'pending')
                print(f"{indent}└─ {name} [{state}]")
            
            # Show pipeline executed
            print("\n📊 Pipeline Executed:")
            print("-"*40)
            for i, node in enumerate(sorted_nodes, 1):
                print(f"{i}. {node['function_block']['name']}: {node['function_block']['description']}")
        
        # Create summary
        summary_path = output_dir / "mock_benchmark_summary.json"
        summary = {
            "test_type": "mock_llm",
            "timestamp": timestamp,
            "user_request": user_request,
            "input_data": str(test_file),
            "total_nodes": result['total_nodes'],
            "completed_nodes": result['completed_nodes'],
            "failed_nodes": result['failed_nodes'],
            "execution_time_seconds": (end_time - start_time).total_seconds(),
            "tree_id": tree_data.get('id') if tree_file.exists() else None,
            "pipeline": [node['function_block']['name'] for node in sorted_nodes] if tree_file.exists() else []
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📄 Summary saved: {summary_path}")
        
        return result, output_dir


def run_clustering_benchmark_openai():
    """Run clustering benchmark with real OpenAI GPT-4o-mini."""
    print("\n" + "="*80)
    print("CLUSTERING BENCHMARK - OPENAI GPT-4O-MINI")
    print("="*80)
    
    # Check for API key
    env_file = Path(".env")
    if not env_file.exists():
        env_file = Path("ragomics_agent_local/.env")
    
    if not env_file.exists():
        print("❌ No .env file found with OpenAI API key")
        print("   Please create .env with OPENAI_API_KEY=your-key")
        return None, None
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in .env file")
        return None, None
    
    print("✅ OpenAI API key loaded")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_test_output_dir() / f"clustering_openai_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output Directory: {output_dir}")
    
    # Create test data
    test_file = create_zebrafish_test_data(output_dir)
    print(f"📊 Test Data: {test_file}")
    print(f"   Size: {test_file.stat().st_size / 1024:.1f} KB")
    
    # Create main agent with real LLM
    main_agent = MainAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Mock node executor to avoid actual Docker execution
    with patch.object(main_agent.node_executor, 'execute_node') as mock_execute:
        def mock_execution(node, tree, input_path, output_base_dir):
            # Create proper output structure
            from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
            tree_manager = AnalysisTreeManager(tree)
            
            # Get nodes directory - corrected structure (no tree_ prefix)
            tree_dir = output_base_dir / tree.id
            nodes_dir = tree_dir / "nodes"
            nodes_dir.mkdir(parents=True, exist_ok=True)
            
            # Create node directory
            node_paths = tree_manager.create_node_directory(node.id, nodes_dir)
            
            # Create job directory
            job_dir = tree_manager.create_job_directory(node.id, node_paths['node_dir'])
            
            # Simulate execution
            start_time = datetime.now()
            
            # Create output
            output_file = job_dir / "output" / "_node_anndata.h5ad"
            if Path(input_path).exists():
                import shutil
                if Path(input_path).is_file():
                    shutil.copy2(input_path, output_file)
                else:
                    for f in Path(input_path).glob("*.h5ad"):
                        shutil.copy2(f, output_file)
                        break
            else:
                output_file.touch()
            
            # Write logs
            (job_dir / "logs" / "stdout.txt").write_text(
                f"Executing {node.function_block.name}\\n" +
                f"Input: {input_path}\\n" +
                f"Processing...\\n" +
                f"Success!"
            )
            (job_dir / "logs" / "stderr.txt").write_text("")
            
            end_time = datetime.now()
            
            # Save execution summary
            tree_manager.save_job_execution_summary(
                job_dir=job_dir,
                node_id=node.id,
                state="success",
                start_time=start_time,
                end_time=end_time,
                input_path=str(input_path),
                output_path=str(job_dir / "output"),
                exit_code=0
            )
            
            # Copy outputs to node
            tree_manager.copy_outputs_to_node(job_dir, node_paths['node_dir'])
            
            return (NodeState.COMPLETED, str(node_paths['node_dir'] / "outputs"))
        
        mock_execute.side_effect = mock_execution
        
        # Run analysis
        print("\n🚀 Running OpenAI Analysis...")
        user_request = """
        Your job is to benchmark different clustering methods on the given zebrafish dataset.
        Apply quality control (filter cells with < 200 genes), normalization (log-transform),
        PCA dimensionality reduction, and then test Leiden clustering at resolutions 0.5 and 1.0.
        Calculate ARI metrics comparing to the ground truth cell_type labels.
        """
        
        start_time = datetime.now()
        
        try:
            result = main_agent.run_analysis(
                input_data_path=test_file,
                user_request=user_request,
                output_dir=output_dir,
                max_nodes=6,  # Allow more nodes for real LLM
                max_children=1,
                verbose=True
            )
            end_time = datetime.now()
            
            print("\n" + "-"*80)
            print("📈 OPENAI RESULTS")
            print("-"*80)
            print(f"✅ Total Nodes: {result['total_nodes']}")
            print(f"✅ Completed: {result['completed_nodes']}")
            print(f"❌ Failed: {result['failed_nodes']}")
            print(f"⏱️ Execution Time: {(end_time - start_time).total_seconds():.2f} seconds")
            
            # Load and display tree structure
            tree_file = Path(result['tree_file'])
            if tree_file.exists():
                with open(tree_file) as f:
                    tree_data = json.load(f)
                
                print("\n🌲 Analysis Tree:")
                print("-"*40)
                
                # Use tree manager for visualization
                tree_manager = AnalysisTreeManager()
                tree_manager.tree = main_agent.tree_manager.tree
                visualization = tree_manager.get_tree_visualization()
                print(visualization)
                
                # Show generated function blocks
                print("\n🔧 Generated Function Blocks:")
                print("-"*40)
                nodes = tree_data.get('nodes', {})
                sorted_nodes = sorted(nodes.values(), key=lambda x: x.get('level', 0))
                
                for i, node in enumerate(sorted_nodes, 1):
                    fb = node['function_block']
                    print(f"\n{i}. {fb['name']}:")
                    print(f"   Type: {fb.get('type', 'python')}")
                    print(f"   Description: {fb.get('description', 'N/A')}")
                    if fb.get('code'):
                        code_preview = fb['code'][:200] + "..." if len(fb['code']) > 200 else fb['code']
                        print(f"   Code Preview: {code_preview}")
            
            # Create summary
            summary_path = output_dir / "openai_benchmark_summary.json"
            summary = {
                "test_type": "openai_gpt-4o-mini",
                "timestamp": timestamp,
                "user_request": user_request,
                "input_data": str(test_file),
                "total_nodes": result['total_nodes'],
                "completed_nodes": result['completed_nodes'],
                "failed_nodes": result['failed_nodes'],
                "execution_time_seconds": (end_time - start_time).total_seconds(),
                "tree_id": tree_data.get('id') if tree_file.exists() else None,
                "pipeline": [node['function_block']['name'] for node in sorted_nodes] if tree_file.exists() else [],
                "llm_model": "gpt-4o-mini"
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n📄 Summary saved: {summary_path}")
            
            return result, output_dir
            
        except Exception as e:
            print(f"\n❌ Error during OpenAI execution: {e}")
            import traceback
            traceback.print_exc()
            return None, output_dir


def compare_results(mock_result, mock_dir, openai_result, openai_dir):
    """Compare mock and OpenAI results."""
    print("\n" + "="*80)
    print("COMPARISON: MOCK vs OPENAI")
    print("="*80)
    
    if not mock_result or not openai_result:
        print("❌ Cannot compare - missing results")
        return
    
    # Load summaries
    mock_summary_file = mock_dir / "mock_benchmark_summary.json"
    openai_summary_file = openai_dir / "openai_benchmark_summary.json" if openai_dir else None
    
    with open(mock_summary_file) as f:
        mock_summary = json.load(f)
    
    if openai_summary_file and openai_summary_file.exists():
        with open(openai_summary_file) as f:
            openai_summary = json.load(f)
    else:
        openai_summary = {}
    
    print("\n📊 Metrics Comparison:")
    print("-"*40)
    print(f"{'Metric':<25} {'Mock':<15} {'OpenAI':<15}")
    print("-"*40)
    print(f"{'Total Nodes':<25} {mock_summary['total_nodes']:<15} {openai_summary.get('total_nodes', 'N/A'):<15}")
    print(f"{'Completed Nodes':<25} {mock_summary['completed_nodes']:<15} {openai_summary.get('completed_nodes', 'N/A'):<15}")
    print(f"{'Failed Nodes':<25} {mock_summary['failed_nodes']:<15} {openai_summary.get('failed_nodes', 'N/A'):<15}")
    print(f"{'Execution Time (s)':<25} {mock_summary['execution_time_seconds']:.2f}{'':>13} {openai_summary.get('execution_time_seconds', 0):.2f}{'':>13}")
    
    print("\n📋 Pipeline Comparison:")
    print("-"*40)
    mock_pipeline = mock_summary['pipeline']
    openai_pipeline = openai_summary.get('pipeline', [])
    
    max_len = max(len(mock_pipeline), len(openai_pipeline))
    print(f"{'Step':<5} {'Mock':<35} {'OpenAI':<35}")
    print("-"*75)
    
    for i in range(max_len):
        mock_step = mock_pipeline[i] if i < len(mock_pipeline) else "-"
        openai_step = openai_pipeline[i] if i < len(openai_pipeline) else "-"
        print(f"{i+1:<5} {mock_step:<35} {openai_step:<35}")
    
    print("\n📈 Summary:")
    print("-"*40)
    print(f"Mock used predefined pipeline with {len(mock_pipeline)} steps")
    if openai_pipeline:
        print(f"OpenAI generated custom pipeline with {len(openai_pipeline)} steps")
        print(f"OpenAI took {openai_summary.get('execution_time_seconds', 0) - mock_summary['execution_time_seconds']:.2f}s longer (includes LLM calls)")
    else:
        print("OpenAI results not available")


if __name__ == "__main__":
    print("\n🧪 RAGOMICS CLUSTERING BENCHMARK TEST")
    print("="*80)
    
    # Run mock benchmark
    mock_result, mock_dir = run_clustering_benchmark_mock()
    
    # Run OpenAI benchmark
    openai_result, openai_dir = run_clustering_benchmark_openai()
    
    # Compare results
    if mock_result:
        compare_results(mock_result, mock_dir, openai_result, openai_dir)
    
    print("\n✅ BENCHMARK COMPLETE")
    print(f"📁 Mock outputs: {mock_dir}")
    if openai_dir:
        print(f"📁 OpenAI outputs: {openai_dir}")