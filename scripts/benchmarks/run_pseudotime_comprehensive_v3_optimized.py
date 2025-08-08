#!/usr/bin/env python3
"""Run comprehensive pseudotime benchmark with optimized context for Claude."""

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
    print("COMPREHENSIVE PSEUDOTIME BENCHMARK V3 (OPTIMIZED) WITH CLAUDE BUG FIXER")
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
    output_dir = Path(f"test_outputs/pseudotime_comprehensive_v3_{timestamp}")
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
    
    # Define comprehensive user request with simpler structure
    user_request = """
Create a pseudotime analysis workflow:

1. Load zebrafish.h5ad and preprocess (normalize_total, log1p, highly_variable_genes, PCA, neighbors, UMAP)

2. Run DPT pseudotime:
   - Use sc.tl.paga(adata, groups='Cell_type')
   - Set root cell: adata.uns['iroot'] = 0
   - Run sc.tl.dpt(adata)
   - Store: adata.obs['dpt_pseudotime'] = adata.obs['dpt_pseudotime']

3. Run simple graph-based pseudotime:
   - Use networkx shortest paths from root cell to all others
   - Store in adata.obs['graph_pseudotime']

4. Compute correlations between all pseudotime methods

5. Plot UMAP colored by each pseudotime method

IMPORTANT: Handle ALL errors gracefully. If a package is missing, skip that method.
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
    
    # Configure Claude service for better rate limit handling
    if hasattr(agent, 'claude_service') and agent.claude_service:
        # Add rate limit handling
        agent.claude_service.max_retries = 3
        agent.claude_service.retry_delay = 5.0  # Wait 5 seconds between retries
    
    # Run analysis with optimized parameters
    print("\n" + "-"*60)
    print("Starting comprehensive pseudotime analysis...")
    print("-"*60 + "\n")
    
    start_time = time.time()
    
    result = agent.run_analysis(
        input_data_path=str(input_dir / "zebrafish.h5ad"),
        user_request=user_request,
        output_dir=str(output_dir),
        max_nodes=10,  # Reduced from 15
        max_children=1,  # Reduced from 2 to create linear workflow
        max_debug_trials=3,  # Reduced from 4
        max_iterations=20,  # Reduced from 30
        generation_mode="mixed",
        verbose=True,
        # Claude bug fixer parameters - optimized for rate limits
        max_claude_turns=3,  # Reduced from 5
        max_claude_cost=0.50,  # Moderate limit
        max_claude_tokens=100000  # Reduced to avoid rate limits
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
            'bug_fixer_type': 'claude',
            'optimized': True
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
        results_dir = output_dir / analysis_id
        if results_dir.exists():
            print(f"✓ Results directory found")
            
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
                
                # Check final output for pseudotime columns
                if output_files:
                    last_output = sorted(output_files)[-1]
                    try:
                        import scanpy as sc
                        adata = sc.read_h5ad(last_output)
                        
                        print(f"\n  Final AnnData:")
                        print(f"    - Shape: {adata.shape}")
                        
                        # Check for pseudotime columns
                        pseudotime_cols = [col for col in adata.obs.columns if 'pseudotime' in col]
                        if pseudotime_cols:
                            print(f"    - Pseudotime columns: {pseudotime_cols}")
                            for col in pseudotime_cols:
                                non_null = adata.obs[col].notna().sum()
                                print(f"      - {col}: {non_null}/{len(adata)} cells")
                        else:
                            print("    - No pseudotime columns found")
                            
                    except Exception as e:
                        print(f"  ! Could not read final AnnData: {e}")
    
    # Check for Claude bug fixer usage
    claude_logs = list(output_dir.rglob("*/claude_debugging/*"))
    if claude_logs:
        print(f"\n✓ Claude bug fixer was used ({len(claude_logs)} times)")
        # Count successful fixes
        successful_fixes = sum(1 for log in claude_logs if "fixed" in str(log))
        print(f"  - Successful fixes: {successful_fixes}")


def write_detailed_report(output_dir: Path, result: Dict, elapsed_time: float):
    """Write a detailed markdown report."""
    
    report_path = output_dir / "COMPREHENSIVE_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Pseudotime Benchmark Report (Optimized)\n\n")
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
        
        f.write("## Configuration (Optimized)\n\n")
        f.write("- Bug Fixer: Claude (Haiku)\n")
        f.write("- LLM Model: GPT-4o-mini\n")
        f.write("- Max Nodes: 10 (reduced)\n")
        f.write("- Max Children: 1 (linear workflow)\n")
        f.write("- Max Debug Trials: 3 (reduced)\n")
        f.write("- Claude Max Turns: 3 (reduced)\n")
        f.write("- Claude Max Tokens: 100,000 (reduced)\n")
        f.write("- Rate Limit Handling: Enabled\n\n")
        
        f.write("## Optimizations Applied\n\n")
        f.write("1. Reduced context size sent to Claude\n")
        f.write("2. Limited file content and error message lengths\n")
        f.write("3. Linear workflow to minimize complexity\n")
        f.write("4. Added retry logic for rate limits\n")
        f.write("5. Simplified user request structure\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Claude Bug Fixer Integration**: ")
        if any(Path(output_dir).rglob("*/claude_debugging/*")):
            f.write("✅ Successfully integrated and used\n")
        else:
            f.write("⚠️  Not used in this run\n")
        
        f.write("2. **Pseudotime Methods**: Check output files for implemented methods\n")
        f.write("3. **Rate Limit Issues**: Check logs for 429 errors\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review the analysis_tree.json for workflow structure\n")
        f.write("2. Check Claude debugging logs for insights\n")
        f.write("3. Examine output AnnData files for pseudotime results\n")
        f.write("4. Monitor rate limit errors and adjust parameters if needed\n")
    
    print(f"\n✓ Detailed report written to: {report_path}")


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Add a small delay to avoid immediate rate limits
    print("Waiting 10 seconds before starting to avoid rate limits...")
    time.sleep(10)
    
    # Run the benchmark
    success = run_comprehensive_pseudotime_benchmark()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)