#!/usr/bin/env python3
"""Test script for clustering benchmark with zebrafish data."""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.agents import MainAgent
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


def main():
    """Run clustering benchmark test."""
    
    # User request for clustering benchmark
    user_request = """Your job is to benchmark different clustering methods on the given dataset. 
    Process scRNA-seq data. Calculate UMAP visualization first with different parameters. 
    Then process the single-cell genomics data. Run at least five clustering methods, 
    and calculate multiple metrics for each clustering method, better based on the 
    ground-truth cell type key provided in the cell meta data. Save the metrics 
    results to anndata object, and output to outputs/."""
    
    # Input data
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    
    # Check if input exists
    if not input_data.exists():
        print(f"Error: Input data not found at {input_data}")
        print("Please ensure the zebrafish.h5ad file is in the data directory")
        sys.exit(1)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_outputs") / "clustering" / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Clustering Benchmark Test")
    print("="*60)
    print(f"Input data: {input_data}")
    print(f"Output directory: {output_dir}")
    print()
    print("User request:")
    print(user_request)
    print("="*60)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try to read from .env file
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    
    if not api_key:
        print("\nError: OpenAI API key not found!")
        print("Please set OPENAI_API_KEY environment variable or create .env file")
        sys.exit(1)
    
    # Create agent
    print("\n1. Initializing MainAgent...")
    agent = MainAgent(openai_api_key=api_key)
    
    # Validate environment
    print("\n2. Validating environment...")
    validation = agent.validate_environment()
    
    all_valid = True
    for component, status in validation.items():
        status_str = "✓" if status else "✗"
        print(f"   {status_str} {component}")
        if not status:
            all_valid = False
    
    if not all_valid:
        print("\nWarning: Some components are missing.")
        print("Docker may not be available or images may not be built.")
        print("Continuing anyway...")
    
    # Run analysis
    print("\n3. Starting analysis with GPT-4o-mini...")
    print("   This will:")
    print("   - Generate function blocks for clustering benchmark")
    print("   - Execute preprocessing")
    print("   - Run multiple clustering methods")
    print("   - Calculate metrics")
    print("   - Save results")
    
    try:
        result = agent.run_analysis(
            input_data_path=input_data,
            user_request=user_request,
            output_dir=output_dir,
            max_nodes=10,  # Allow up to 10 nodes for comprehensive analysis
            max_children=3,
            max_debug_trials=2,
            generation_mode="mixed",
            llm_model="gpt-4o-mini",  # Use GPT-4o-mini as requested
            verbose=True
        )
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        print(f"Output directory: {result['output_dir']}")
        print(f"Tree ID: {result['tree_id']}")
        print(f"Total nodes: {result['total_nodes']}")
        print(f"Completed nodes: {result['completed_nodes']}")
        print(f"Failed nodes: {result['failed_nodes']}")
        print(f"Analysis tree saved to: {result['tree_file']}")
        
        # Show results
        if result.get('results'):
            print("\nExecution Results:")
            for node_id, node_result in result['results'].items():
                status_icon = "✓" if "completed" in node_result['state'] else "✗"
                print(f"  {status_icon} {node_result['name']}: {node_result['state']}")
                if node_result.get('output'):
                    print(f"    Output: {node_result['output']}")
                if node_result.get('error'):
                    print(f"    Error: {node_result['error']}")
        
        # Check if the analysis achieved the goals
        print("\n" + "="*60)
        print("Goal Achievement Check:")
        print("="*60)
        
        completed = result['completed_nodes']
        total = result['total_nodes']
        
        if completed == 0:
            print("✗ No nodes were executed successfully")
            print("  The analysis failed to start. Check the error messages above.")
        elif completed < total:
            print(f"⚠ Partially completed: {completed}/{total} nodes executed")
            print("  Some analysis steps failed. Check the outputs for partial results.")
        else:
            print(f"✓ All {completed} nodes executed successfully!")
            print("  The clustering benchmark should be complete.")
        
        # Provide guidance on results
        print("\nTo inspect the results:")
        print(f"1. Check the output directory: {result['output_dir']}")
        print("2. Look for clustering metrics in the final output .h5ad file")
        print("3. Review the analysis tree for the complete workflow")
        
        return 0 if completed > 0 else 1
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())