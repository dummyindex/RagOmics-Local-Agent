#!/usr/bin/env python3
"""Test pseudotime benchmark with multiple methods."""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.models import GenerationMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pseudotime_benchmark():
    """Run pseudotime benchmark test."""
    
    # Test data path
    test_data = Path(__file__).parent.parent / "test_data" / "zebrafish.h5ad"
    if not test_data.exists():
        logger.error(f"Test data not found: {test_data}")
        return False
    
    # User request for pseudotime benchmark
    user_request = """Preprocess dataset if not processed yet.
Run DPT + PAGA pseudotime (Python, Scanpy) – store in adata.obs['dpt']
Run Slingshot (R) – save output to slingshot_pseudotime.csv, load into adata.obs['slingshot']
Run Palantir (Python) – store in adata.obs['palantir']
Run Monocle 3 (R) – save output to monocle3_pseudotime.csv, load into adata.obs['monocle3']
Run scFates (Python) – store in adata.obs['scFates']
Compute metrics (Kendall, Spearman, MAE) for all methods vs. ground-truth. Store in adata.uns['pseudotime_metrics']
Plot pseudotime on UMAP, one subplot per method
Plot metric comparisons, one subplot per metric (bar plots of all methods)"""
    
    # Create workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = Path("test_outputs") / f"pseudotime_benchmark_{timestamp}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test data
    test_data_copy = workspace_dir / "test_data" / "zebrafish.h5ad"
    test_data_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_data, test_data_copy)
    
    # Initialize agent
    logger.info("Initializing MainAgent for pseudotime benchmark")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return None
    
    agent = MainAgent(openai_api_key=api_key, llm_model="gpt-4o")
    
    try:
        # Run analysis
        logger.info(f"Starting analysis with request: {user_request[:100]}...")
        
        results = agent.run_analysis(
            input_data_path=str(test_data_copy),
            user_request=user_request,
            output_dir=workspace_dir,
            max_nodes=30,  # Allow enough nodes for all methods
            max_children=1,  # Only one branch per node
            max_debug_trials=3,
            max_iterations=20,
            generation_mode="only_new",
            llm_model="gpt-4o",
            verbose=True
        )
        
        # Get tree for analysis
        tree = agent.tree_manager.tree
        
        # Final analysis
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        
        # Get node counts
        completed_nodes = [n for n in tree.nodes.values() if n.state.value == "completed"]
        failed_nodes = [n for n in tree.nodes.values() if n.state.value == "failed"]
        
        # Check expected outputs
        expected_methods = ['dpt', 'slingshot', 'palantir', 'monocle3', 'scfates']
        benchmark_results = {
            "completed_nodes": len(completed_nodes),
            "failed_nodes": len(failed_nodes),
            "methods_completed": [],
            "has_metrics": False,
            "has_plots": False
        }
        
        # Check which methods completed
        for node in completed_nodes:
            name = node.function_block.name.lower()
            for method in expected_methods:
                if method in name:
                    benchmark_results["methods_completed"].append(method)
                    break
            
            if "metric" in name:
                benchmark_results["has_metrics"] = True
            if "plot" in name or "viz" in name:
                benchmark_results["has_plots"] = True
        
        # Check for output files
        for node in completed_nodes:
            if node.output_data_id:
                output_dir = Path(node.output_data_id)
                if output_dir.exists():
                    # Check for figures
                    figures_dir = output_dir.parent / "figures"
                    if figures_dir.exists():
                        figures = list(figures_dir.glob("*.png"))
                        if figures:
                            logger.info(f"Found {len(figures)} figures in {figures_dir}")
                            benchmark_results["has_plots"] = True
                    
                    # Check for AnnData
                    adata_file = output_dir / "_node_anndata.h5ad"
                    if adata_file.exists():
                        logger.info(f"Found AnnData at {adata_file}")
        
        # Summary
        logger.info(f"\nCompleted methods: {benchmark_results['methods_completed']}")
        logger.info(f"Has metrics: {benchmark_results['has_metrics']}")
        logger.info(f"Has plots: {benchmark_results['has_plots']}")
        
        # Save results
        results_file = workspace_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        
        # Generate tree visualization
        try:
            tree_md = agent.tree_manager.tree.to_markdown()
            tree_file = workspace_dir / "analysis_tree.md"
            with open(tree_file, 'w') as f:
                f.write(tree_md)
        except Exception as e:
            logger.warning(f"Could not generate tree markdown: {e}")
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logger.info("Starting pseudotime benchmark test")
    results = run_pseudotime_benchmark()
    
    if results:
        logger.info("\nBenchmark completed!")
        if len(results["methods_completed"]) >= 3:
            logger.info("✅ SUCCESS: Multiple pseudotime methods completed")
        else:
            logger.info("⚠️  PARTIAL: Only some methods completed")
    else:
        logger.error("❌ FAILED: Benchmark did not complete")