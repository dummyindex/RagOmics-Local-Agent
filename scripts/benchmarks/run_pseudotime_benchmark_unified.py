#!/usr/bin/env python
"""Run pseudotime benchmark with unified bug fixer."""

import os
import sys
import argparse
import asyncio
from pathlib import Path
import shutil
import time
from datetime import datetime
import json

# Add parent directory to path for imports (scripts/benchmarks -> project root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ragomics_agent_local.agents.main_agent import MainAgent


def setup_test_environment(output_base: Path, model_name: str) -> tuple[Path, Path]:
    """Set up test directories and copy input data."""
    # Create unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = model_name.replace("-", "_").replace(".", "_")
    output_dir = output_base / f"pseudotime_unified_{model_suffix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy zebrafish data
    input_dir = output_dir / "input"
    input_dir.mkdir(exist_ok=True)
    
    zebrafish_data = Path("data/zebrafish/_node_anndata.h5ad")
    if zebrafish_data.exists():
        shutil.copy(zebrafish_data, input_dir / "_node_anndata.h5ad")
        print(f"✓ Copied zebrafish data to {input_dir}")
    else:
        print(f"✗ Zebrafish data not found at {zebrafish_data}")
        sys.exit(1)
    
    return input_dir, output_dir


def create_benchmark_request() -> str:
    """Create the pseudotime benchmark request."""
    return """I need a comprehensive Python-only pseudotime analysis. Please:

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
   - One subplot per method using adata.obs
   - Save as figures/pseudotime_comparison.png

9. Plot metric comparisons:
   - One subplot per metric
   - Bar plots with methods on x-axis, scores on y-axis
   - Save as figures/metric_comparison.png

IMPORTANT: Skip methods if packages are not available. Use only Python."""


def write_benchmark_report(output_dir: Path, result: dict, model_name: str, duration: float):
    """Write detailed benchmark report."""
    report_path = output_dir / "UNIFIED_BENCHMARK_REPORT.md"
    
    # Extract statistics
    success = result.get('completed_nodes', 0) > 0
    bug_fixer_stats = result.get('claude_usage', {}) if 'claude_usage' in result else {}
    
    report = f"""# Unified Bug Fixer Pseudotime Benchmark Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {model_name}
**Duration**: {duration:.1f} seconds ({duration/60:.1f} minutes)
**Success**: {'✅ Yes' if success else '❌ No'}

## Summary

- Total nodes: {result.get('total_nodes', 0)}
- Completed nodes: {result.get('completed_nodes', 0)}
- Failed nodes: {result.get('failed_nodes', 0)}
- Tree ID: {result.get('tree_id', 'N/A')}

## Bug Fixer Statistics

- Model: {bug_fixer_stats.get('model', model_name)}
- Total Cost: ${bug_fixer_stats.get('total_cost', 0):.4f}
- Total Tokens: {bug_fixer_stats.get('total_tokens', 0):,}
- Sessions: {bug_fixer_stats.get('sessions', 0)}
- Success Rate: {bug_fixer_stats.get('success_rate', 0)*100:.1f}%

## Expected Outputs

1. **Preprocessing**: Normalized, log-transformed data with HVGs, PCA, and UMAP
2. **DPT Pseudotime**: Diffusion pseudotime in `adata.obs['dpt_pseudotime']`
3. **Palantir**: Trajectory inference (if package available)
4. **scFates**: Tree-based pseudotime (if package available)
5. **CellRank**: Markov chain pseudotime (if package available)
6. **Graph Pseudotime**: Simple shortest-path based pseudotime
7. **Metrics**: Correlation metrics comparing methods
8. **Visualizations**: UMAP plots and metric comparisons

## Output Files

- Analysis tree: `{output_dir}/analysis_tree.json`
- Debug logs: `{output_dir}/*/agent_tasks/unified_bug_fixer/`
- Output data: `{output_dir}/*/output/_node_anndata.h5ad`
- Figures: `{output_dir}/*/output/figures/`

## Model Performance

### {model_name}
- Average turns per fix: {bug_fixer_stats.get('average_turns', 0):.1f} (if available)
- Cost efficiency: ${bug_fixer_stats.get('total_cost', 0) / max(1, bug_fixer_stats.get('sessions', 1)):.4f} per session
"""
    
    report_path.write_text(report)
    print(f"\n✓ Detailed report written to: {report_path}")


async def run_benchmark(model_name: str, api_key: str = None):
    """Run benchmark with specified model."""
    print(f"\n{'='*60}")
    print(f"Running Pseudotime Benchmark with Unified Bug Fixer")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Set up environment
    output_base = Path("test_outputs")
    input_dir, output_dir = setup_test_environment(output_base, model_name)
    
    # Validate environment
    print("\n✓ Initializing Main Agent with unified bug fixer...")
    
    # Determine API keys
    if model_name.startswith('gpt'):
        openai_key = api_key or os.getenv('OPENAI_API_KEY')
        claude_key = None
    else:
        openai_key = None
        claude_key = api_key or os.getenv('ANTHROPIC_API_KEY')
    
    # Create agent
    agent = MainAgent(
        openai_api_key=openai_key,
        claude_api_key=claude_key,
        bug_fixer_type="unified",
        bug_fixer_model=model_name,
        llm_model="gpt-4o-mini"  # For orchestration
    )
    
    # Validate environment
    env_check = agent.validate_environment()
    print(f"✓ Environment validation: {env_check}")
    
    # Run benchmark
    print("\n" + "-"*60)
    print("Starting pseudotime benchmark analysis...")
    print("-"*60 + "\n")
    
    start_time = time.time()
    
    try:
        result = agent.process_request(
            request=create_benchmark_request(),
            input_dir=input_dir,
            output_dir=output_dir,
            max_nodes=12,
            max_debug_trials=3,
            verbose=True
        )
        
        duration = time.time() - start_time
        
        # Display results
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"Success: {result.get('completed_nodes', 0) > 0}")
        print(f"Analysis ID: {result.get('tree_id', 'N/A')}")
        print(f"Time elapsed: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Tree nodes: {result.get('total_nodes', 0)}")
        print(f"Completed nodes: {result.get('completed_nodes', 0)}")
        print(f"Failed nodes: {result.get('failed_nodes', 0)}")
        
        # Write report
        write_benchmark_report(output_dir, result, model_name, duration)
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n✗ Benchmark failed after {duration:.1f} seconds: {e}")
        raise
    
    print(f"\n✓ Results saved to: {output_dir}")
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run pseudotime benchmark with unified bug fixer")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name (e.g., gpt-4, claude-3-5-sonnet-20241022)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models"
    )
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022"
        ]
        
        print("Running benchmark comparison across models...")
        for model in models:
            try:
                # Check if we have the required API key
                if model.startswith('gpt') and not os.getenv('OPENAI_API_KEY'):
                    print(f"\n⚠️  Skipping {model} - no OPENAI_API_KEY")
                    continue
                elif model.startswith('claude') and not os.getenv('ANTHROPIC_API_KEY'):
                    print(f"\n⚠️  Skipping {model} - no ANTHROPIC_API_KEY")
                    continue
                
                asyncio.run(run_benchmark(model))
                
            except Exception as e:
                print(f"\n✗ Failed to run benchmark with {model}: {e}")
    else:
        # Run single model
        asyncio.run(run_benchmark(args.model))


if __name__ == "__main__":
    main()
