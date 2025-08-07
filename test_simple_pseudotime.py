#!/usr/bin/env python3
"""Simple pseudotime test focusing on getting basic methods working."""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.models import GenerationMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simple_pseudotime():
    """Run simplified pseudotime test."""
    
    # Test data path
    test_data = Path(__file__).parent / "test_data" / "zebrafish.h5ad"
    if not test_data.exists():
        logger.error(f"Test data not found: {test_data}")
        return False
    
    # Simplified request focusing on core functionality
    user_request = """1. Quality control: filter cells with min_genes=200, filter genes with min_cells=3
2. Normalize data using total-count normalization and log1p transformation
3. Calculate 50 principal components
4. Run DPT + PAGA pseudotime analysis and store result in adata.obs['dpt_pseudotime']
5. Create UMAP visualization colored by pseudotime"""
    
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
            max_nodes=10,  # Limit nodes
            max_children=3,
            max_debug_trials=3,
            max_iterations=10,
            generation_mode="mixed",
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
        
        logger.info(f"Total nodes: {results['total_nodes']}")
        logger.info(f"Completed: {results['completed_nodes']}")
        logger.info(f"Failed: {results['failed_nodes']}")
        
        # Check if DPT was completed
        has_dpt = any('dpt' in node.function_block.name.lower() for node in completed_nodes)
        
        # Save summary
        summary = {
            "total_nodes": results['total_nodes'],
            "completed_nodes": results['completed_nodes'],
            "failed_nodes": results['failed_nodes'],
            "has_dpt": has_dpt,
            "node_names": [n.function_block.name for n in completed_nodes]
        }
        
        summary_file = workspace_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nSummary saved to {summary_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    logger.info("Starting simple pseudotime test")
    results = run_simple_pseudotime()
    
    if results and results['completed_nodes'] >= 4:
        logger.info("\n✅ SUCCESS: Basic pseudotime pipeline completed")
    else:
        logger.error("\n❌ FAILED: Pipeline did not complete")