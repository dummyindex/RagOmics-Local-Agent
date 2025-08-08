#!/usr/bin/env python3
"""Run pseudotime analysis with intentional errors to test Claude bug fixer."""

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


def run_pseudotime_with_errors():
    """Run pseudotime analysis with errors to trigger Claude bug fixer."""
    
    print("\n" + "="*80)
    print("PSEUDOTIME ANALYSIS WITH ERRORS TO TEST CLAUDE BUG FIXER")
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
    output_dir = Path(f"test_outputs/pseudotime_errors_{timestamp}")
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
    
    # Define request that will cause errors requiring Claude to fix
    user_request = """
Create a pseudotime analysis workflow with these specific requirements:

1. Load zebrafish.h5ad and preprocess:
   - Use adata.X = adata.X.toarray() if sparse
   - Normalize with sc.pp.normalize_total(adata, target_sum=10000)
   - Apply log transformation: sc.pp.log1p(adata)
   - Find highly variable genes with wrong parameter: sc.pp.highly_variable_genes(adata, n_top=2000)
   
2. Run Palantir pseudotime (will fail without the package):
   - import palantir
   - ms_data = palantir.utils.run_magic(adata)
   - Set start cell: start_cell = adata.obs_names[0]
   - Run: pr_res = palantir.core.run_palantir(ms_data, start_cell)
   - Store: adata.obs['palantir_pseudotime'] = pr_res.pseudotime
   
3. Run scFates pseudotime (will fail without the package):
   - import scFates as scf
   - scf.pp.find_root(adata, root_adata.obs_names[0])
   - scf.tl.pseudotime(adata, n_map=1, use_rep="X_pca")
   - Store: adata.obs['scfates_pseudotime'] = adata.obs['pseudotime']
   
4. Run a custom pseudotime with intentional error:
   - Use wrong import: from sklearn.neighbors import NearestNeighbors (should be from sklearn.neighbors import NearestNeighbors)
   - Compute custom pseudotime using wrong attribute: nn.fit(adata.X_pca)
   - Store in adata.obs['custom_pseudotime']

IMPORTANT: Let the errors happen naturally and use Claude bug fixer to resolve them.
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
    print("Starting pseudotime analysis with errors...")
    print("-"*60 + "\n")
    
    start_time = time.time()
    
    result = agent.run_analysis(
        input_data_path=str(input_dir / "zebrafish.h5ad"),
        user_request=user_request,
        output_dir=str(output_dir),
        max_nodes=8,
        max_children=1,
        max_debug_trials=4,  # Allow multiple attempts
        max_iterations=15,
        generation_mode="only_new",  # Force new implementations
        verbose=True,
        # Claude bug fixer parameters
        max_claude_turns=5,
        max_claude_cost=0.50,
        max_claude_tokens=100000
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
    
    # Verify Claude usage
    print("\n" + "-"*60)
    print("Claude Bug Fixer Usage")
    print("-"*60)
    
    # Count Claude debugging attempts
    claude_debug_dirs = list(output_dir.rglob("*/claude_debugging/*"))
    print(f"✓ Claude debugging sessions: {len(claude_debug_dirs)}")
    
    # Check for successful fixes
    successful_fixes = 0
    for debug_dir in claude_debug_dirs:
        report_files = list(debug_dir.glob("*_report.json"))
        for report_file in report_files:
            try:
                with open(report_file) as f:
                    report = json.load(f)
                    if report.get('success'):
                        successful_fixes += 1
                        print(f"  ✓ Fixed: {report.get('node_name', 'unknown')}")
            except:
                pass
    
    print(f"✓ Successful fixes: {successful_fixes}")
    
    # Log Claude usage
    if hasattr(agent, 'claude_service') and agent.claude_service:
        print(f"\n✓ Total Claude API cost: ${agent.claude_service.total_cost:.4f}")
        print(f"✓ Total Claude tokens used: {agent.claude_service.total_tokens_used}")
    
    print("\n✓ Test complete!")
    print(f"✓ Results saved to: {output_dir}")
    
    # Write summary
    summary_path = output_dir / "claude_test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            'claude_sessions': len(claude_debug_dirs),
            'successful_fixes': successful_fixes,
            'total_cost': getattr(agent.claude_service, 'total_cost', 0) if hasattr(agent, 'claude_service') else 0,
            'total_tokens': getattr(agent.claude_service, 'total_tokens_used', 0) if hasattr(agent, 'claude_service') else 0,
            **result
        }, f, indent=2)
    
    return result.get('success', False)


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Wait to avoid rate limits
    print("Waiting 10 seconds before starting...")
    time.sleep(10)
    
    # Run the test
    success = run_pseudotime_with_errors()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)