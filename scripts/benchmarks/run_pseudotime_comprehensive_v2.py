#!/usr/bin/env python3
"""Run comprehensive pseudotime benchmark with better prompting."""

import os
import sys
import time
import json
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.config import config
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


def run_comprehensive_pseudotime_benchmark():
    """Run comprehensive pseudotime analysis with all methods."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PSEUDOTIME BENCHMARK V2 WITH CLAUDE BUG FIXER")
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
    output_dir = Path(f"test_outputs/pseudotime_comprehensive_v2_{timestamp}")
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
    
    # Define comprehensive user request with explicit node structure
    user_request = """
Please create a comprehensive pseudotime analysis workflow with these SEPARATE nodes:

Node 1: "preprocess_data" - Load zebrafish.h5ad and perform preprocessing:
- normalize_total with target_sum=1e4
- log1p transform
- Save as _node_anndata.h5ad

Node 2: "find_hvg_and_reduce" - Find highly variable genes and reduce dimensions:
- highly_variable_genes with n_top_genes=2000
- PCA with 50 components
- Save as _node_anndata.h5ad

Node 3: "compute_neighbors_umap" - Compute neighbors and UMAP:
- neighbors with n_neighbors=10, n_pcs=40
- UMAP embedding
- Save as _node_anndata.h5ad

Node 4: "run_dpt_paga" - Run DPT with PAGA:
- Compute PAGA (sc.tl.paga)
- Select root cell (first cell or based on known marker)
- Run diffusion pseudotime (sc.tl.dpt)
- Store result in adata.obs['dpt_pseudotime']
- Save as _node_anndata.h5ad

Node 5: "run_simple_pseudotime" - Run simple pseudotime methods:
- Implement graph-based pseudotime using shortest paths from a root cell
- Store in adata.obs['graph_pseudotime']
- Implement PCA-based pseudotime (project onto first PC)
- Store in adata.obs['pca_pseudotime']
- Save as _node_anndata.h5ad

Node 6: "run_palantir" - Run Palantir if available:
- Try to import palantir, if not available, skip this method
- If available, run palantir with appropriate start cell
- Store result in adata.obs['palantir_pseudotime']
- Save as _node_anndata.h5ad

Node 7: "compute_correlations" - Compute pseudotime correlations:
- For each pair of pseudotime methods, compute Spearman correlation
- Store correlation matrix in adata.uns['pseudotime_correlations']
- Print correlation summary
- Save as _node_anndata.h5ad

Node 8: "plot_pseudotime_comparison" - Create comparison plots:
- Create figure with subplots showing UMAP colored by each pseudotime
- Use consistent viridis colormap
- Save as figures/pseudotime_comparison_umap.png

Node 9: "plot_correlation_heatmap" - Plot correlation heatmap:
- Create heatmap of pseudotime correlations
- Annotate with correlation values
- Save as figures/pseudotime_correlation_heatmap.png

IMPORTANT: Create each node separately. Use Python for all nodes. Handle missing packages gracefully.
"""
    
    # Initialize main agent with Claude bug fixer
    print("\n✓ Initializing Main Agent with Claude bug fixer...")
    agent = MainAgent(
        openai_api_key=openai_key,
        claude_api_key=claude_key,
        bug_fixer_type="claude",
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
        max_nodes=15,  # Allow sufficient nodes
        max_children=2,  # Allow some branching
        max_debug_trials=4,
        max_iterations=30,  # Reasonable iterations
        generation_mode="mixed",  # Allow both new and existing blocks
        verbose=True,
        # Claude bug fixer parameters
        max_claude_turns=5,
        max_claude_cost=1.00,  # Increased from 0.10 for comprehensive test
        max_claude_tokens=300000  # Increased from 30000 to handle complex debugging
    )
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Success: {result.get('success', False)}")
    print(f"Analysis ID: {result.get('analysis_id', 'N/A')}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
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
    
    # Write detailed report
    write_detailed_report(output_dir, result, elapsed_time)
    
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
                        
                        # Check for correlations
                        if 'pseudotime_correlations' in adata.uns:
                            print(f"    - Pseudotime correlations computed")
                            
                    except Exception as e:
                        print(f"  ! Could not read final AnnData: {e}")
        else:
            print(f"✗ Results directory not found")
    
    # Check for Claude bug fixer logs
    claude_logs = list(Path(output_dir).rglob("*/agent_tasks/claude_bug_fixer/*"))
    if claude_logs:
        print(f"\n✓ Claude bug fixer was used ({len(claude_logs)} times)")
        # Count successful fixes
        successful_fixes = 0
        for log_dir in claude_logs:
            report_files = list(log_dir.parent.parent.glob("claude_debugging/*_report.json"))
            for report_file in report_files:
                try:
                    with open(report_file) as f:
                        report = json.load(f)
                        if report.get('success'):
                            successful_fixes += 1
                except:
                    pass
        print(f"  - {successful_fixes} successful fixes")


def write_detailed_report(output_dir: Path, result: Dict, elapsed_time: float):
    """Write a detailed markdown report."""
    
    report_path = output_dir / "COMPREHENSIVE_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Pseudotime Benchmark Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Duration**: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)\n")
        f.write(f"**Success**: {'✅ Yes' if result.get('success') else '❌ No'}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total nodes: {result.get('tree_nodes', 0)}\n")
        f.write(f"- Completed nodes: {result.get('completed_nodes', 0)}\n")
        f.write(f"- Failed nodes: {result.get('failed_nodes', 0)}\n")
        f.write(f"- Analysis ID: {result.get('analysis_id', 'N/A')}\n\n")
        
        if result.get('error'):
            f.write(f"## Error\n\n{result['error']}\n\n")
        
        f.write("## Configuration\n\n")
        f.write("- Bug Fixer: Claude (Haiku)\n")
        f.write("- LLM Model: GPT-4o-mini\n")
        f.write("- Max Nodes: 15\n")
        f.write("- Max Debug Trials: 4\n")
        f.write("- Claude Max Turns: 5\n")
        f.write("- Claude Max Cost: $0.10\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Claude Bug Fixer Integration**: ")
        if any(Path(output_dir).rglob("*/claude_debugging/*")):
            f.write("✅ Successfully integrated and used\n")
        else:
            f.write("⚠️  Not used in this run\n")
        
        f.write("2. **Pseudotime Methods Implemented**: ")
        # List methods based on what we find
        f.write("(Check output files for details)\n")
        
        f.write("3. **Visualizations Created**: ")
        figures = list(Path(output_dir).rglob("*/figures/*.png"))
        f.write(f"{len(figures)} figures\n\n")
        
        if figures:
            f.write("### Figures\n\n")
            for fig in sorted(figures):
                f.write(f"- {fig.name}\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("1. Check the analysis_tree.json for the complete workflow\n")
        f.write("2. Review Claude debugging logs for bug fixing insights\n")
        f.write("3. Examine the final AnnData object for all pseudotime results\n")
    
    print(f"\n✓ Detailed report written to: {report_path}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the benchmark
    success = run_comprehensive_pseudotime_benchmark()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)