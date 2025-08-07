#!/usr/bin/env python3
"""Monitor pseudotime benchmark execution in real-time."""

import os
import sys
import time
import json
from pathlib import Path
import shutil
from datetime import datetime
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.config import config


class ExecutionMonitor:
    """Monitor execution progress in real-time."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        self.nodes_seen = set()
        self.monitoring = True
        
    def monitor_loop(self):
        """Monitor the execution directory for changes."""
        while self.monitoring:
            try:
                # Check for tree updates
                tree_file = self.output_dir / "results" / "analysis_tree.json"
                if tree_file.exists():
                    with open(tree_file) as f:
                        tree = json.load(f)
                        
                    # Check for new nodes
                    for node_id, node in tree.get('nodes', {}).items():
                        if node_id not in self.nodes_seen:
                            self.nodes_seen.add(node_id)
                            elapsed = time.time() - self.start_time
                            print(f"\n[{elapsed:.1f}s] New node: {node['function_block']['name']}")
                            print(f"         State: {node['state']}")
                            if node.get('error'):
                                print(f"         Error: {node['error']}")
                                
                # Check for execution logs
                nodes_dir = self.output_dir / "results" / "c0d915d5-1b2a-41de-8e27-1e6ba6207f95" / "nodes"
                if nodes_dir.exists():
                    for node_dir in nodes_dir.iterdir():
                        if node_dir.is_dir():
                            log_file = node_dir / "outputs" / "execution.log"
                            if log_file.exists() and log_file.stat().st_mtime > self.start_time:
                                # New or updated log
                                pass
                                
            except Exception as e:
                pass
                
            time.sleep(1)
            
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False


def run_monitored_benchmark():
    """Run benchmark with real-time monitoring."""
    
    print("\n" + "="*80)
    print("MONITORED PSEUDOTIME BENCHMARK")
    print("="*80 + "\n")
    
    # Setup directories
    output_dir = Path("test_outputs/pseudotime_monitor")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    input_dir = output_dir / "input"
    input_dir.mkdir()
    
    # Copy data
    shutil.copy("test_data/zebrafish.h5ad", input_dir / "zebrafish.h5ad")
    print(f"✓ Copied zebrafish data")
    
    # User request
    user_request = """
Run a comprehensive pseudotime analysis benchmark on the zebrafish dataset:

1. Preprocess the dataset:
   - Use Scanpy to normalize (normalize_total), log transform (log1p), 
   - Find highly variable genes (highly_variable_genes)
   - Compute PCA, build neighbor graph, and compute UMAP

2. Run DPT pseudotime analysis:
   - Use sc.tl.dpt() to compute diffusion pseudotime
   - Store result in adata.obs['dpt_pseudotime']

3. Run Palantir pseudotime analysis:
   - Use the PCA embedding from preprocessing
   - Compute Palantir pseudotime and store in adata.obs['palantir_pseudotime']

4. Create visualization:
   - Plot UMAP colored by each pseudotime method
   - Save as pseudotime_comparison.png

Start with preprocessing, then run each pseudotime method sequentially.
"""
    
    # Configure
    config.function_block_timeout = 600
    config.execution_timeout = 3600
    
    # Start monitor
    monitor = ExecutionMonitor(output_dir)
    monitor_thread = threading.Thread(target=monitor.monitor_loop)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Create agent and run
    print("\nStarting benchmark execution...\n")
    main_agent = MainAgent()
    
    try:
        result = main_agent.run_analysis(
            input_data_path=str(input_dir / "zebrafish.h5ad"),
            user_request=user_request,
            output_dir=str(output_dir / "results"),
            max_nodes=10,
            max_children=1,
            max_debug_trials=3,
            max_iterations=30,
            generation_mode="mixed",
            verbose=True
        )
        
        duration = time.time() - monitor.start_time
        print(f"\n\nExecution completed in {duration:.1f} seconds")
        
        # Summary
        print("\nExecution Summary:")
        print(f"  Total nodes: {result.get('total_nodes', 0)}")
        print(f"  Completed: {result.get('completed_nodes', 0)}")
        print(f"  Failed: {result.get('failed_nodes', 0)}")
        
        # Check outputs
        results_dir = output_dir / "results"
        if results_dir.exists():
            print("\nOutput files:")
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    if file.endswith(('.png', '.h5ad', '.csv')):
                        rel_path = os.path.relpath(os.path.join(root, file), results_dir)
                        print(f"  - {rel_path}")
                        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        monitor.stop()
        
    return output_dir


if __name__ == "__main__":
    output_dir = run_monitored_benchmark()
    print(f"\nResults saved to: {output_dir}")