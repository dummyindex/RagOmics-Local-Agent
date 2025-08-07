#!/usr/bin/env python3
"""Test Python-only pseudotime benchmark with zebrafish data."""

import os
import sys
import time
import json
from pathlib import Path
import shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.config import config


def run_python_pseudotime_benchmark():
    """Run pseudotime benchmark with Python methods only."""
    
    print("\n" + "="*80)
    print("PYTHON-ONLY PSEUDOTIME BENCHMARK TEST")
    print("="*80 + "\n")
    
    # Create test output directory
    output_dir = Path("test_outputs/pseudotime_python_benchmark")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Prepare input data
    input_dir = output_dir / "input"
    input_dir.mkdir()
    
    # Copy zebrafish data
    zebrafish_path = Path("../test_data/zebrafish.h5ad")
    if not zebrafish_path.exists():
        print(f"Error: {zebrafish_path} not found!")
        return False
        
    shutil.copy(zebrafish_path, input_dir / "zebrafish.h5ad")
    print(f"✓ Copied zebrafish data to {input_dir}")
    
    # Create user request with Python methods only
    user_request = """
Run a comprehensive pseudotime analysis benchmark on the zebrafish dataset using multiple Python methods.

CRITICAL: CREATE SEPARATE NODES FOR EACH TASK. DO NOT COMBINE MULTIPLE METHODS IN ONE NODE.

The workflow should have these SEPARATE nodes:

Node 1: Preprocess dataset (if not already processed)
   - Use Scanpy: normalize_total → log1p → highly_variable_genes → PCA → neighbors → UMAP
   - Save preprocessed data as _node_anndata.h5ad

Node 2: Run DPT (depends on Node 1)
   - Compute diffusion pseudotime using sc.tl.dpt()
   - Auto-select root cell if needed: adata.uns['iroot'] = np.argmax(adata.obs['n_genes_by_counts'])
   - Store result in adata.obs['dpt_pseudotime']
   - Save output as _node_anndata.h5ad

Node 3: Run PAGA (depends on Node 1)
   - Compute PAGA trajectory using sc.tl.paga()
   - Store connectivity in adata.uns['paga']
   - Save output as _node_anndata.h5ad

Node 4: Run Palantir using Scanpy's external API (depends on Node 1)
   - Install: pip install palantir
   - Use scanpy.external.tl.palantir()
   - Store result in adata.obs['palantir_pseudotime']
   - Save output as _node_anndata.h5ad

Node 5: Run scFates (depends on Node 1)
   - Learn trajectory tree and compute pseudotime
   - Store in adata.obs['scfates_pseudotime']
   - Save output as _node_anndata.h5ad

Node 6: Run CellRank (depends on Node 1)
   - Use VelocityKernel or ConnectivityKernel
   - Compute terminal states and absorption probabilities
   - Store in adata.obs['cellrank_pseudotime'] or appropriate key
   - Save output as _node_anndata.h5ad

Node 7: Merge all results (depends on Nodes 2-6)
   - Read all pseudotime results from previous nodes
   - Combine into single anndata object
   - Save merged data as _node_anndata.h5ad

Node 8: Compute evaluation metrics (depends on Node 7)
   - Compare each method's pseudotime
   - Use Kendall's tau, Spearman's rho, and MAE between methods
   - Store all results in adata.uns['pseudotime_metrics']
   - Save output as _node_anndata.h5ad

Node 9: Generate comparison plots (depends on Node 8)
   - Plot pseudotime over UMAP for each method
   - Create metric comparison bar plots
   - Save all figures to figures/ directory

Use the zebrafish.h5ad file as input.
"""
    
    # Configure agent
    print("\nConfiguring main agent...")
    
    # Set reasonable timeouts
    config.function_block_timeout = 600  # 10 minutes per block
    config.execution_timeout = 3600  # 1 hour total
    
    # Create main agent with API key to enable orchestrator
    main_agent = MainAgent(openai_api_key=config.openai_api_key)
    
    # Run the benchmark
    print("\nStarting pseudotime benchmark...")
    print(f"Start time: {datetime.now()}")
    
    start_time = time.time()
    
    try:
        # Process the request
        result = main_agent.run_analysis(
            input_data_path=str(input_dir / "zebrafish.h5ad"),
            user_request=user_request,
            output_dir=str(output_dir / "results"),
            max_nodes=20,
            max_children=1,  # Linear workflow
            max_debug_trials=4,
            max_iterations=50,  # Allow many iterations for debugging
            generation_mode="mixed",
            verbose=True
        )
        
        duration = time.time() - start_time
        print(f"\nBenchmark completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        # Check results
        if result.get('completed_nodes', 0) > 0:
            print("\n✓ Benchmark completed successfully!")
            
            # List output files
            results_dir = output_dir / "results"
            if results_dir.exists():
                print("\nOutput files:")
                for root, dirs, files in os.walk(results_dir):
                    level = root.replace(str(results_dir), '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
                        
            # Check for expected outputs
            print("\nChecking for expected outputs:")
            expected_files = [
                "pseudotime_umap.png",
                "metric_comparison.png",
                "processed_data.h5ad"
            ]
            
            for expected in expected_files:
                found = False
                for root, dirs, files in os.walk(results_dir):
                    if expected in files:
                        found = True
                        print(f"  ✓ Found {expected}")
                        break
                if not found:
                    print(f"  ✗ Missing {expected}")
                    
        else:
            print(f"\n✗ Benchmark failed: Completed {result.get('completed_nodes', 0)} nodes")
            
        # Save execution log
        log_file = output_dir / "execution_log.json"
        with open(log_file, 'w') as f:
            json.dump({
                'request': user_request,
                'duration_seconds': duration,
                'total_nodes': result.get('total_nodes', 0),
                'completed_nodes': result.get('completed_nodes', 0),
                'failed_nodes': result.get('failed_nodes', 0),
                'tree_file': result.get('tree_file'),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
            
        print(f"\nExecution log saved to: {log_file}")
        
        success = result.get('completed_nodes', 0) > 0 and result.get('failed_nodes', 0) == 0
        return success
        
    except Exception as e:
        print(f"\n✗ Exception during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_failures(output_dir):
    """Analyze any failures in the benchmark run."""
    
    print("\n" + "="*80)
    print("FAILURE ANALYSIS")
    print("="*80 + "\n")
    
    # Look for error logs
    results_dir = Path(output_dir) / "results"
    if not results_dir.exists():
        print("No results directory found")
        return
        
    # Check each node's output
    for node_dir in results_dir.glob("node_*"):
        if node_dir.is_dir():
            print(f"\nAnalyzing {node_dir.name}:")
            
            # Check for error files
            error_file = node_dir / "error.txt"
            if error_file.exists():
                print(f"  ✗ Error found:")
                with open(error_file) as f:
                    error_content = f.read()
                    print("    " + error_content.replace("\n", "\n    "))
                    
            # Check for output files
            output_files = list(node_dir.glob("*"))
            print(f"  Files: {[f.name for f in output_files]}")
            
            # Check logs
            log_file = node_dir / "execution.log"
            if log_file.exists():
                print(f"  Execution log exists")
                

if __name__ == "__main__":
    # Run the benchmark
    success = run_python_pseudotime_benchmark()
    
    # Analyze failures if any
    if not success:
        analyze_failures("test_outputs/pseudotime_python_benchmark")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)