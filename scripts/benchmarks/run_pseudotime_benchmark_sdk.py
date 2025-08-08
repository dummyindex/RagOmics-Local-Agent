#!/usr/bin/env python3
"""Run Python-only pseudotime benchmark with Claude Code SDK bug fixer."""

import os
import sys
import time
import json
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict

# Add parent directory to path for imports (scripts/benchmarks -> project root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.config import config
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


def run_pseudotime_benchmark():
    """Run pseudotime benchmark with Claude Code SDK bug fixer."""
    
    print("\n" + "="*80)
    print("PYTHON-ONLY PSEUDOTIME BENCHMARK WITH CLAUDE CODE SDK")
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
    
    # Check Claude Code CLI
    import subprocess
    cli_commands = ['claude-code', 'claude']  # Try both possible names
    cli_found = False
    
    for cmd in cli_commands:
        try:
            result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                cli_found = True
                print(f"✓ Found Claude Code CLI as '{cmd}'")
                break
        except FileNotFoundError:
            continue
    
    if not cli_found:
        print("Error: Claude Code CLI not found. Please install: npm install -g @anthropic-ai/claude-code")
        return False
    
    # Create test output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"test_outputs/pseudotime_sdk_{timestamp}")
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
    
    # Define pseudotime benchmark request
    user_request = """
1. Preprocess dataset if not already processed
   - Use Scanpy: normalize_total → log1p → highly_variable_genes → PCA → neighbors → UMAP

2. Run DPT + PAGA (Scanpy)
   - Compute pseudotime using sc.tl.dpt()
   - Store result in adata.obs['dpt']

3. Run Palantir
   - Use palantir package if available, otherwise skip
   - Store result in adata.obs['palantir']

4. Run scFates
   - Use scFates package if available, otherwise skip
   - Learn trajectory tree and compute pseudotime
   - Store in adata.obs['scFates']

5. Run CellRank
   - Use cellrank package if available, otherwise skip
   - Use PseudotimeKernel or VelocityKernel
   - Store pseudotime or absorption probabilities in adata.obs['cellrank']

6. Run simple graph-based pseudotime
   - Use networkx to compute shortest paths from a root cell
   - Store result in adata.obs['graph_pseudotime']

7. Compute evaluation metrics:
   - Compare each method's pseudotime with ground-truth labels (if available)
   - Use Kendall's tau, Spearman's rho, and MAE
   - Store all results in adata.uns['pseudotime_metrics']

8. Plot pseudotime over UMAP:
   - One subplot per method using adata.obs[method]
   - Save as figures/pseudotime_comparison.png

9. Plot metric comparisons:
   - One subplot per metric
   - Bar plots with methods on x-axis, scores on y-axis
   - Save as figures/metric_comparison.png

IMPORTANT: Skip methods if packages are not available. Use only Python.
"""
    
    # Initialize main agent with Claude Code SDK bug fixer
    print("\n✓ Initializing Main Agent with Claude Code SDK bug fixer...")
    agent = MainAgent(
        openai_api_key=openai_key,
        claude_api_key=claude_key,
        bug_fixer_type="claude_code_sdk",  # Use SDK bug fixer
        llm_model="gpt-4o-mini"
    )
    
    # Validate environment
    validation = agent.validate_environment()
    print(f"✓ Environment validation: {validation}")
    
    # Run analysis
    print("\n" + "-"*60)
    print("Starting pseudotime benchmark analysis...")
    print("-"*60 + "\n")
    
    start_time = time.time()
    
    result = agent.run_analysis(
        input_data_path=str(input_dir / "zebrafish.h5ad"),
        user_request=user_request,
        output_dir=str(output_dir),
        max_nodes=12,
        max_children=1,  # Linear workflow
        max_debug_trials=3,
        max_iterations=20,
        generation_mode="mixed",
        verbose=True
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
            'bug_fixer_type': 'claude_code_sdk'
        }, f, indent=2)
    
    # Verify outputs
    print("\n" + "-"*60)
    print("Verifying outputs...")
    print("-"*60)
    verify_outputs(output_dir, result.get('analysis_id'))
    
    # Log Claude Code SDK usage
    if 'claude_usage' in result:
        usage = result['claude_usage']
        print(f"\n✓ Claude Code SDK Statistics:")
        print(f"  - Total cost: ${usage.get('total_cost', 0):.4f}")
        print(f"  - Total tokens: {usage.get('total_tokens', 0)}")
        print(f"  - Sessions: {usage.get('sessions', 0)}")
        print(f"  - Success rate: {usage.get('success_rate', 0):.1%}")
    
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
        results_dir = output_dir / analysis_id
        if results_dir.exists():
            print(f"✓ Results directory found")
            
            # Count nodes
            nodes_dir = results_dir / "nodes"
            if nodes_dir.exists():
                node_count = len(list(nodes_dir.glob("node_*")))
                print(f"  - {node_count} nodes created")
                
                # Check for SDK debugging logs
                sdk_logs = list(nodes_dir.glob("*/agent_tasks/claude_code_sdk/*/debug_history.json"))
                if sdk_logs:
                    print(f"  - {len(sdk_logs)} Claude Code SDK debugging sessions")
                
                # Check for outputs
                output_files = list(nodes_dir.glob("*/outputs/_node_anndata.h5ad"))
                print(f"  - {len(output_files)} output AnnData files")
                
                # Check for figures
                figure_files = list(nodes_dir.glob("*/outputs/figures/*.png"))
                print(f"  - {len(figure_files)} figure files")
                
                # Check final output for pseudotime columns
                if output_files:
                    last_output = sorted(output_files)[-1]
                    try:
                        import scanpy as sc
                        adata = sc.read_h5ad(last_output)
                        
                        print(f"\n  Final AnnData:")
                        print(f"    - Shape: {adata.shape}")
                        
                        # Check for pseudotime columns
                        pseudotime_cols = [col for col in adata.obs.columns if 'pseudotime' in col or col in ['dpt', 'palantir', 'scFates', 'cellrank']]
                        if pseudotime_cols:
                            print(f"    - Pseudotime columns: {pseudotime_cols}")
                            for col in pseudotime_cols:
                                non_null = adata.obs[col].notna().sum()
                                print(f"      - {col}: {non_null}/{len(adata)} cells")
                        else:
                            print("    - No pseudotime columns found")
                        
                        # Check for metrics
                        if 'pseudotime_metrics' in adata.uns:
                            print(f"    - Pseudotime metrics computed")
                            metrics = adata.uns['pseudotime_metrics']
                            if isinstance(metrics, dict):
                                for metric, values in metrics.items():
                                    print(f"      - {metric}: {values}")
                            
                    except Exception as e:
                        print(f"  ! Could not read final AnnData: {e}")


def write_detailed_report(output_dir: Path, result: Dict, elapsed_time: float):
    """Write a detailed markdown report."""
    
    report_path = output_dir / "PSEUDOTIME_BENCHMARK_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# Pseudotime Benchmark Report (Claude Code SDK)\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Duration**: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)\n")
        f.write(f"**Success**: {'✅ Yes' if result.get('success') else '❌ No'}\n")
        f.write(f"**Bug Fixer**: Claude Code SDK\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total nodes: {result.get('tree_nodes', 0)}\n")
        f.write(f"- Completed nodes: {result.get('completed_nodes', 0)}\n")
        f.write(f"- Failed nodes: {result.get('failed_nodes', 0)}\n")
        f.write(f"- Analysis ID: {result.get('analysis_id', 'N/A')}\n\n")
        
        if result.get('error'):
            f.write(f"## Error\n\n{result['error']}\n\n")
        
        f.write("## Configuration\n\n")
        f.write("- Bug Fixer: Claude Code SDK\n")
        f.write("- LLM Model: GPT-4o-mini\n")
        f.write("- Max Nodes: 12\n")
        f.write("- Max Debug Trials: 3\n")
        f.write("- Generation Mode: mixed\n\n")
        
        if 'claude_usage' in result:
            usage = result['claude_usage']
            f.write("## Claude Code SDK Usage\n\n")
            f.write(f"- Total Cost: ${usage.get('total_cost', 0):.4f}\n")
            f.write(f"- Total Tokens: {usage.get('total_tokens', 0):,}\n")
            f.write(f"- Debugging Sessions: {usage.get('sessions', 0)}\n")
            f.write(f"- Success Rate: {usage.get('success_rate', 0):.1%}\n\n")
        
        f.write("## Expected Outputs\n\n")
        f.write("1. **Preprocessing**: Normalized, log-transformed data with HVGs, PCA, and UMAP\n")
        f.write("2. **DPT Pseudotime**: Diffusion pseudotime in `adata.obs['dpt']`\n")
        f.write("3. **Additional Methods**: Palantir, scFates, CellRank (if packages available)\n")
        f.write("4. **Graph Pseudotime**: Simple graph-based pseudotime\n")
        f.write("5. **Metrics**: Correlation metrics in `adata.uns['pseudotime_metrics']`\n")
        f.write("6. **Visualizations**: UMAP plots and metric comparisons\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Check the analysis_tree.json for workflow structure\n")
        f.write("2. Review SDK debugging logs in agent_tasks/claude_code_sdk/\n")
        f.write("3. Examine output AnnData files for pseudotime results\n")
        f.write("4. Verify figures in outputs/figures/\n")
    
    print(f"\n✓ Detailed report written to: {report_path}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the benchmark
    success = run_pseudotime_benchmark()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)