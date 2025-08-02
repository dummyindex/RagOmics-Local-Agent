#!/usr/bin/env python3
"""Final test for clustering benchmark with correct configuration."""

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
    print("Final Clustering Benchmark Test with GPT-4o-mini")
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
    
    # User request with correct cell type column
    user_request = """Your job is to benchmark different clustering methods on the given zebrafish dataset. 
    
    The data has a ground truth cell type column called 'Cell_type' (with capital C).
    
    Please:
    1. Preprocess the data if needed (normalization, highly variable genes, PCA)
    2. Calculate UMAP visualization with at least 3 different parameter sets
    3. Run at least 5 clustering methods (leiden with different resolutions, louvain, kmeans, hierarchical, spectral)
    4. Calculate multiple metrics for each clustering method including:
       - ARI (Adjusted Rand Index) against Cell_type
       - NMI (Normalized Mutual Information) against Cell_type
       - Silhouette score
       - Calinski-Harabasz score
    5. Save all metrics results to adata.uns['clustering_metrics']
    6. Save figures showing UMAP plots colored by different clustering results
    7. Create a summary table of all metrics
    
    Use 'Cell_type' as the ground truth key for metrics calculation."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("test_outputs") / "clustering" / f"final_{timestamp}"
    
    print(f"\n2. Input data: {input_data}")
    print(f"   - Shape: 4181 cells × 16940 genes")
    print(f"   - Ground truth: Cell_type column with 12 cell types")
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
    print("   - Handle errors with bug fixer agent")
    print()
    
    try:
        # Run the analysis
        results = agent.run_analysis(
            input_data_path=input_data,
            user_request=user_request,
            output_dir=output_dir,
            max_nodes=3,  # Allow multiple nodes
            max_children=2,
            max_debug_trials=2,  # Allow bug fixing
            generation_mode="only_new",  # Generate new function blocks
            llm_model="gpt-4o-mini",
            verbose=True
        )
        
        # Display results
        print("\n" + "="*60)
        print("Analysis Results")
        print("="*60)
        print(f"✓ Analysis completed!")
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
                if 'error' in node_result and node_result['state'] == 'failed':
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
                            
                            print(f"   Data shape after processing: {adata.shape}")
                            
                            # Check for clustering results
                            clustering_keys = [k for k in adata.obs.columns if 
                                             'leiden' in k or 'louvain' in k or 
                                             'kmeans' in k or 'cluster' in k or
                                             'hierarchical' in k or 'spectral' in k]
                            if clustering_keys:
                                print(f"   Found clustering results: {clustering_keys}")
                            
                            # Check for metrics
                            if 'clustering_metrics' in adata.uns:
                                print("   ✓ Clustering metrics found in adata.uns!")
                                metrics = adata.uns['clustering_metrics']
                                
                                # Display metrics summary
                                print("\n   Clustering Metrics Summary:")
                                print("   " + "-"*40)
                                if isinstance(metrics, dict):
                                    for method, method_metrics in metrics.items():
                                        print(f"   {method}:")
                                        if isinstance(method_metrics, dict):
                                            for metric_name, value in method_metrics.items():
                                                if isinstance(value, (int, float)):
                                                    print(f"     - {metric_name}: {value:.4f}")
                                                else:
                                                    print(f"     - {metric_name}: {value}")
                            else:
                                print("   Note: No clustering_metrics found in adata.uns")
                            
                            # Check for figures
                            figures_dir = Path(node_result['output']).parent / "figures"
                            if figures_dir.exists():
                                figures = list(figures_dir.glob("*.png"))
                                if figures:
                                    print(f"\n   Generated figures ({len(figures)}):")
                                    for fig in figures[:5]:  # Show first 5
                                        print(f"     - {fig.name}")
                                
                        except Exception as e:
                            print(f"   Could not load output data: {e}")
        
        # Success summary
        if results['completed_nodes'] > 0:
            print("\n" + "="*60)
            print("✓ SUCCESS: Clustering benchmark completed!")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("✗ PARTIAL SUCCESS: Some nodes failed")
            print("="*60)
            return 1
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())