#!/usr/bin/env python3
"""Run comprehensive pseudotime benchmark with all methods."""

import os
import sys
import time
import json
from pathlib import Path
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.config import config
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


def run_comprehensive_pseudotime_benchmark():
    """Run comprehensive pseudotime analysis with all methods."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PSEUDOTIME BENCHMARK WITH CLAUDE BUG FIXER")
    print("="*80 + "\n")
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    claude_key = os.getenv('CLAUDE_API_KEY')
    
    if not openai_key:
        print("Error: OPENAI_API_KEY not set")
        return False
    if not claude_key:
        print("Error: CLAUDE_API_KEY not set")
        return False
    
    # Create test output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"test_outputs/pseudotime_comprehensive_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare input data
    input_dir = output_dir / "input"
    input_dir.mkdir()
    
    # Copy zebrafish data
    zebrafish_path = Path("test_data/zebrafish.h5ad")
    if not zebrafish_path.exists():
        print(f"Error: {zebrafish_path} not found!")
        return False
        
    shutil.copy(zebrafish_path, input_dir / "zebrafish.h5ad")
    print(f"✓ Copied zebrafish data to {input_dir}")
    
    # Define comprehensive user request
    user_request = """
Perform comprehensive pseudotime analysis comparing multiple methods on the zebrafish dataset:

1. Preprocess dataset if not already processed
   - Use Scanpy: normalize_total → log1p → highly_variable_genes → PCA → neighbors → UMAP
   - Save preprocessed data with all embeddings

2. Run DPT + PAGA (Scanpy)
   - First compute PAGA (sc.tl.paga)
   - Then compute pseudotime using sc.tl.dpt() with root cell selection
   - Store result in adata.obs['dpt_pseudotime']

3. Run Palantir
   - Install palantir package if needed
   - Select start cell and run palantir
   - Store result in adata.obs['palantir_pseudotime']

4. Run scFates (if available for Python)
   - Install scfates if needed
   - Learn trajectory tree and compute pseudotime
   - Store in adata.obs['scfates_pseudotime']
   - If not available, skip this method

5. Run CellRank
   - Install cellrank if needed
   - Use PseudotimeKernel based on existing pseudotime or VelocityKernel if RNA velocity available
   - Compute terminal states and fate probabilities
   - Store main pseudotime in adata.obs['cellrank_pseudotime']

6. Run simple pseudotime methods as alternatives
   - Implement a simple graph-based pseudotime using shortest paths
   - Store in adata.obs['graph_pseudotime']
   - Implement PCA-based pseudotime projection
   - Store in adata.obs['pca_pseudotime']

7. Compute evaluation metrics:
   - For each pseudotime method, compute correlation with other methods
   - Use Kendall's tau and Spearman's rho
   - If ground truth time points available in adata.obs, compare against those
   - Store all correlation results in adata.uns['pseudotime_correlations']

8. Plot pseudotime over UMAP:
   - Create a figure with subplots for each pseudotime method
   - Color cells by pseudotime values
   - Use consistent color scale (viridis)
   - Save as 'pseudotime_comparison_umap.png'

9. Plot correlation heatmap:
   - Create correlation matrix heatmap between all methods
   - Annotate with correlation values
   - Save as 'pseudotime_correlation_heatmap.png'

Important notes:
- Create separate function blocks for each major step
- Use Python for all analysis steps
- Handle missing packages gracefully - install if needed
- If a method fails, log the error but continue with other methods
- Store all intermediate results in the AnnData object
- Save all figures to the figures directory
"""
    
    # Initialize main agent with Claude bug fixer
    print("\n✓ Initializing Main Agent with Claude bug fixer...")
    agent = MainAgent(
        openai_api_key=openai_key,
        claude_api_key=claude_key,
        bug_fixer_type="claude",  # Use Claude bug fixer
        llm_model="gpt-4o-mini"
    )
    
    # Validate environment
    validation = agent.validate_environment()
    print(f"✓ Environment validation: {validation}")
    
    # Run analysis
    print("\n" + "-"*60)
    print("Starting comprehensive pseudotime analysis...")
    print("-"*60 + "\n")
    
    start_time = time.time()
    
    result = agent.run_analysis(
        input_data_path=str(input_dir / "zebrafish.h5ad"),
        user_request=user_request,
        output_dir=str(output_dir),
        max_nodes=25,  # Allow more nodes for comprehensive analysis
        max_children=3,
        max_debug_trials=4,  # As requested
        max_iterations=50,  # Allow sufficient iterations
        generation_mode="only_new",
        verbose=True,
        # Claude bug fixer parameters
        max_claude_turns=5,
        max_claude_cost=0.05,
        max_claude_tokens=20000
    )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Success: {result.get('success', False)}")
    print(f"Analysis ID: {result.get('analysis_id', 'N/A')}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    print(f"Tree nodes: {result.get('tree_nodes', 0)}")
    print(f"Completed nodes: {result.get('completed_nodes', 0)}")
    print(f"Failed nodes: {result.get('failed_nodes', 0)}")
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    # Save analysis summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            **result,
            'elapsed_time': elapsed_time,
            'timestamp': timestamp,
            'bug_fixer_type': 'claude'
        }, f, indent=2)
    
    # Verify outputs
    print("\n" + "-"*60)
    print("Verifying outputs...")
    print("-"*60)
    verify_outputs(output_dir, result.get('analysis_id'))
    
    # Log Claude usage
    if hasattr(agent, 'claude_service') and agent.claude_service:
        print(f"\n✓ Total Claude API cost: ${agent.claude_service.total_cost:.4f}")
        print(f"✓ Total Claude tokens used: {agent.claude_service.total_tokens_used}")
    
    print("\n✓ Analysis complete!")
    print(f"✓ Results saved to: {output_dir}")
    
    return result.get('success', False)


def verify_outputs(output_dir: Path, analysis_id: str):
    """Verify expected outputs exist."""
    
    # Check for analysis tree
    tree_file = output_dir / "analysis_tree.json"
    if tree_file.exists():
        print("✓ Analysis tree found")
        with open(tree_file) as f:
            tree = json.load(f)
            print(f"  - Root node: {tree.get('root_id')}")
    else:
        print("✗ Analysis tree not found")
    
    # Check results directory
    if analysis_id:
        results_dir = output_dir / "results" / analysis_id
        if results_dir.exists():
            print(f"✓ Results directory found: {results_dir}")
            
            # Count nodes
            nodes_dir = results_dir / "nodes"
            if nodes_dir.exists():
                node_count = len(list(nodes_dir.glob("node_*")))
                print(f"  - {node_count} nodes created")
                
                # Check for outputs
                output_files = list(nodes_dir.glob("*/outputs/_node_anndata.h5ad"))
                print(f"  - {len(output_files)} output AnnData files")
                
                # Check for figures
                figure_files = list(nodes_dir.glob("*/outputs/figures/*.png"))
                print(f"  - {len(figure_files)} figure files")
                
                # List figures
                if figure_files:
                    print("\n  Figures created:")
                    for fig in sorted(figure_files):
                        print(f"    - {fig.parent.parent.parent.name}/{fig.name}")
                
                # Check final output
                if output_files:
                    last_output = sorted(output_files)[-1]
                    try:
                        import scanpy as sc
                        adata = sc.read_h5ad(last_output)
                        
                        print(f"\n  Final AnnData:")
                        print(f"    - Shape: {adata.shape}")
                        print(f"    - Layers: {list(adata.layers.keys())}")
                        print(f"    - Obsm: {list(adata.obsm.keys())}")
                        
                        # Check for pseudotime columns
                        pseudotime_cols = [col for col in adata.obs.columns if 'pseudotime' in col]
                        if pseudotime_cols:
                            print(f"    - Pseudotime columns: {pseudotime_cols}")
                        
                        # Check for metrics
                        if 'pseudotime_correlations' in adata.uns:
                            print(f"    - Pseudotime correlations computed")
                        if 'pseudotime_metrics' in adata.uns:
                            print(f"    - Pseudotime metrics computed")
                            
                    except Exception as e:
                        print(f"  ! Could not read final AnnData: {e}")
        else:
            print(f"✗ Results directory not found")
    
    # Check for Claude bug fixer logs
    claude_logs = list(Path(output_dir).rglob("*/agent_tasks/claude_bug_fixer/*"))
    if claude_logs:
        print(f"\n✓ Claude bug fixer was used ({len(claude_logs)} times)")
        for log_dir in claude_logs[:3]:  # Show first 3
            node_name = log_dir.parent.parent.name
            print(f"  - {node_name}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the benchmark
    success = run_comprehensive_pseudotime_benchmark()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)