#!/usr/bin/env python3
"""Test simple pseudotime computation with just preprocessing and DPT."""

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


def run_simple_pseudotime():
    """Run a simple pseudotime test with just DPT."""
    
    # Test data path
    test_data = Path(__file__).parent.parent / "test_data" / "zebrafish.h5ad"
    if not test_data.exists():
        logger.error(f"Test data not found: {test_data}")
        return False
    
    # Simple user request - just DPT
    user_request = """
1. Quality control: filter cells with min_genes=200, filter genes with min_cells=3
2. Normalize data with total counts normalization (target_sum=1e4) and log transform
3. Find highly variable genes (min_mean=0.0125, max_mean=3, min_disp=0.5)
4. Run PCA on highly variable genes
5. Compute neighborhood graph (n_neighbors=10)
6. Run diffusion map
7. Run DPT pseudotime analysis and store result in adata.obs['pseudotime_dpt']
8. Create UMAP visualization colored by pseudotime
"""
    
    # Create workspace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = Path("test_outputs") / f"simple_pseudotime_{timestamp}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test data
    test_data_copy = workspace_dir / "test_data" / "zebrafish.h5ad"
    test_data_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_data, test_data_copy)
    
    # Initialize agent
    logger.info("Initializing MainAgent for simple pseudotime test")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return None
    
    agent = MainAgent(openai_api_key=api_key)
    
    try:
        # Run analysis
        logger.info(f"Starting analysis with request: {user_request[:100]}...")
        
        results = agent.run_analysis(
            input_data_path=str(test_data_copy),
            user_request=user_request,
            output_dir=workspace_dir,
            max_nodes=15,  # Allow enough nodes for preprocessing steps
            max_children=3,
            max_debug_trials=3,
            max_iterations=10,
            generation_mode="only_new",
            llm_model="gpt-4o-mini",
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
        
        # Check results
        results = {
            "completed_nodes": len(completed_nodes),
            "failed_nodes": len(failed_nodes),
            "has_pseudotime": False,
            "has_umap": False
        }
        
        # Check for outputs
        for node in completed_nodes:
            name = node.function_block.name.lower()
            if 'dpt' in name or 'pseudotime' in name:
                results["has_pseudotime"] = True
            if 'umap' in name or 'visualization' in name:
                results["has_umap"] = True
        
        # Save results
        results_file = workspace_dir / "simple_pseudotime_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        logger.info(f"Completed nodes: {results['completed_nodes']}")
        logger.info(f"Failed nodes: {results['failed_nodes']}")
        logger.info(f"Has pseudotime: {results['has_pseudotime']}")
        logger.info(f"Has UMAP: {results['has_umap']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logger.info("Starting simple pseudotime test")
    results = run_simple_pseudotime()
    
    if results:
        logger.info("\nTest completed!")
        if results["has_pseudotime"]:
            logger.info("✅ SUCCESS: Pseudotime computation completed")
        else:
            logger.info("❌ FAILED: Pseudotime not computed")
    else:
        logger.error("❌ FAILED: Test did not complete")