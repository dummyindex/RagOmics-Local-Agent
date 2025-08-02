#!/usr/bin/env python3
"""Test clustering benchmark with full agentic system using GPT-4o-mini."""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.agents import MainAgent
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


def main():
    """Run clustering benchmark using the agentic system."""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return 1
    
    # Initialize main agent
    print("="*60)
    print("Clustering Benchmark Test with GPT-4o-mini")
    print("="*60)
    
    agent = MainAgent(openai_api_key=api_key)
    
    # Validate environment
    print("\n1. Validating environment...")
    validation = agent.validate_environment()
    for component, status in validation.items():
        print(f"  - {component}: {'✓' if status else '✗'}")
    
    # Setup test data
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    if not input_data.exists():
        print(f"\nError: Test data not found at {input_data}")
        return 1
    
    # User request for clustering benchmark
    user_request = """Your job is to benchmark different clustering methods on the given dataset. 
    Process scRNA-seq data. Calculate UMAP visualization first with different parameters. 
    Then process the single-cell genomics data. Run at least five clustering methods, 
    and calculate multiple metrics for each clustering method, better based on the 
    ground-truth cell type key provided in the cell meta data. Save the metrics 
    results to anndata object, and output to outputs/."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_outputs") / "clustering" / f"agent_{timestamp}"
    
    print(f"\n2. Input data: {input_data}")
    print(f"3. Output directory: {output_dir}")
    
    print("\n4. User request:")
    print("-" * 40)
    print(user_request)
    print("-" * 40)
    
    print("\n5. Running analysis with GPT-4o-mini...")
    print("   This will:")
    print("   - Generate function blocks using LLM")
    print("   - Create analysis tree")
    print("   - Execute the analysis pipeline")
    print()
    
    try:
        # Run the analysis
        results = agent.run_analysis(
            input_data_path=input_data,
            user_request=user_request,
            output_dir=output_dir,
            max_nodes=5,  # Limit nodes for testing
            max_children=2,
            max_debug_trials=1,
            generation_mode="only_new",  # Generate new function blocks
            llm_model="gpt-4o-mini",
            verbose=True
        )
        
        # Display results
        print("\n" + "="*60)
        print("Analysis Results")
        print("="*60)
        print(f"✓ Analysis completed successfully!")
        print(f"  - Tree ID: {results['tree_id']}")
        print(f"  - Total nodes: {results['total_nodes']}")
        print(f"  - Completed nodes: {results['completed_nodes']}")
        print(f"  - Failed nodes: {results['failed_nodes']}")
        print(f"  - Output directory: {results['output_dir']}")
        print(f"  - Analysis tree: {results['tree_file']}")
        
        # Display node results
        if results['results']:
            print("\nNode Execution Results:")
            for node_id, node_result in results['results'].items():
                status_symbol = "✓" if node_result['state'] in ['completed', 'completed_after_fix'] else "✗"
                print(f"  {status_symbol} {node_result['name']}: {node_result['state']}")
                if 'output' in node_result:
                    print(f"    Output: {node_result['output']}")
                if 'error' in node_result:
                    print(f"    Error: {node_result['error'][:100]}...")
        
        # Check for clustering metrics
        if results['completed_nodes'] > 0:
            print("\n6. Checking for clustering metrics...")
            # Find the last completed node's output
            for node_result in results['results'].values():
                if node_result.get('state') in ['completed', 'completed_after_fix'] and 'output' in node_result:
                    output_file = Path(node_result['output']) / "output_data.h5ad"
                    if output_file.exists():
                        print(f"   Output data found: {output_file}")
                        
                        # Load and check for metrics
                        try:
                            import scanpy as sc
                            adata = sc.read_h5ad(output_file)
                            
                            # Check for clustering results
                            clustering_keys = [k for k in adata.obs.columns if 
                                             'leiden' in k or 'louvain' in k or 
                                             'kmeans' in k or 'cluster' in k]
                            if clustering_keys:
                                print(f"   Found clustering results: {clustering_keys[:5]}...")
                            
                            # Check for metrics
                            if 'clustering_metrics' in adata.uns:
                                print("   ✓ Clustering metrics found in adata.uns!")
                                metrics = adata.uns['clustering_metrics']
                                print(f"   Number of methods benchmarked: {len(metrics.get('method', []))}")
                            else:
                                print("   Note: No clustering_metrics found in adata.uns")
                                
                        except Exception as e:
                            print(f"   Could not load output data: {e}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())