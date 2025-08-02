#!/usr/bin/env python3
"""Test main agent with scvelo pipeline using mock LLM."""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, GenerationMode, NewFunctionBlock, 
    FunctionBlockType, StaticConfig, Arg
)
from ragomics_agent_local.agents import (
    MainAgent, FunctionSelectorAgent, BugFixerAgent, OrchestratorAgent
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.analysis_tree_management.node_executor import NodeExecutor
from ragomics_agent_local.utils import setup_logger

logger = setup_logger("test_main_agent_scvelo")


class MockFunctionSelector(FunctionSelectorAgent):
    """Mock function selector that returns predefined function blocks for scvelo pipeline."""
    
    def __init__(self):
        super().__init__(llm_service=None)
        self.step_counter = 0
        self.pipeline_steps = self._create_pipeline_steps()
        
    def _create_pipeline_steps(self):
        """Create the scvelo pipeline steps."""
        return [
            # Step 1: Quality Control
            {
                'function_blocks': [self._create_qc_block()],
                'satisfied': False,
                'reasoning': 'Starting with quality control'
            },
            # Step 2: Preprocessing
            {
                'function_blocks': [self._create_preprocessing_block()],
                'satisfied': False,
                'reasoning': 'Preprocessing data for velocity analysis'
            },
            # Step 3: Velocity preparation and pseudotime
            {
                'function_blocks': [
                    self._create_velocity_prep_block(),
                    self._create_pseudotime_block()
                ],
                'satisfied': False,
                'reasoning': 'Preparing for velocity and calculating pseudotime'
            },
            # Step 4: Run scVelo
            {
                'function_blocks': [self._create_scvelo_block()],
                'satisfied': False,
                'reasoning': 'Running RNA velocity analysis'
            },
            # Step 5: Generate plots
            {
                'function_blocks': [
                    self._create_velocity_plots_block(),
                    self._create_analysis_plots_block()
                ],
                'satisfied': True,  # Final step
                'reasoning': 'Generating visualization plots'
            }
        ]
    
    def _mock_select_function_blocks(self, context):
        """Return the next step in the pipeline."""
        # Determine which step we're on based on tree state
        tree = context.get('tree')
        current_node = context.get('current_node')
        
        # If no tree or root node, return first step
        if not tree or not tree.root_node_id:
            self.step_counter = 0
        elif current_node:
            # Determine step based on current node name
            node_name = current_node.function_block.name
            if 'quality_control' in node_name:
                self.step_counter = 1
            elif 'preprocessing' in node_name:
                self.step_counter = 2
            elif 'velocity_prep' in node_name or 'pseudotime' in node_name:
                self.step_counter = 3
            elif 'scvelo' in node_name:
                self.step_counter = 4
            else:
                self.step_counter = min(self.step_counter + 1, len(self.pipeline_steps) - 1)
        
        if self.step_counter < len(self.pipeline_steps):
            result = self.pipeline_steps[self.step_counter]
            return result
        else:
            return {
                'function_blocks': [],
                'satisfied': True,
                'reasoning': 'Pipeline completed'
            }
    
    def _create_qc_block(self):
        """Create quality control function block."""
        config = StaticConfig(
            args=[
                Arg(name="min_genes", value_type="int", description="Minimum genes per cell", 
                    optional=True, default_value=200),
                Arg(name="max_pct_mito", value_type="float", description="Maximum mitochondrial percentage",
                    optional=True, default_value=20.0)
            ],
            description="Quality control for single-cell data",
            tag="qc",
            source="mock"
        )
        
        code = '''
def run(adata, min_genes=200, max_pct_mito=20.0, **kwargs):
    """Perform quality control."""
    import scanpy as sc
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print(f"Initial data shape: {adata.shape}")
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs.pct_counts_mt < max_pct_mito, :]
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=3)
    
    print(f"After QC: {adata.shape}")
    
    # QC plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True, ax=axes[0, :])
    
    axes[1, 0].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.3)
    axes[1, 0].set_xlabel('Total counts')
    axes[1, 0].set_ylabel('Number of genes')
    
    axes[1, 1].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], alpha=0.3)
    axes[1, 1].set_xlabel('Total counts')
    axes[1, 1].set_ylabel('Mitochondrial %')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/qc_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return adata
'''
        
        return NewFunctionBlock(
            name="quality_control_velocity",
            type=FunctionBlockType.PYTHON,
            description="Quality control for velocity analysis",
            code=code,
            requirements="scanpy>=1.9.0\nmatplotlib>=3.6.0\npandas>=2.0.0",
            parameters={"min_genes": 200, "max_pct_mito": 20.0},
            static_config=config
        )
    
    def _create_preprocessing_block(self):
        """Create preprocessing function block."""
        config = StaticConfig(
            args=[],
            description="Preprocessing for velocity analysis",
            tag="preprocessing",
            source="mock"
        )
        
        code = '''
def run(adata, **kwargs):
    """Preprocess data for velocity analysis."""
    import scanpy as sc
    import scvelo as scv
    import numpy as np
    
    print("Preprocessing for velocity analysis...")
    
    # Store raw counts for velocity
    adata.raw = adata.copy()
    
    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable].copy()
    
    # Scale data
    sc.pp.scale(adata, max_value=10)
    
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=40)
    
    # UMAP
    sc.tl.umap(adata)
    
    # Clustering
    sc.tl.leiden(adata, resolution=0.6)
    
    print(f"Preprocessed data shape: {adata.shape}")
    
    # Create synthetic spliced/unspliced layers if not present
    if 'spliced' not in adata.layers:
        print("Creating synthetic spliced/unspliced layers for demonstration...")
        adata.layers['spliced'] = adata.X.copy()
        adata.layers['unspliced'] = adata.X.copy() * np.random.uniform(0.2, 0.4, adata.shape)
    
    return adata
'''
        
        return NewFunctionBlock(
            name="preprocessing_velocity",
            type=FunctionBlockType.PYTHON,
            description="Preprocessing for velocity analysis",
            code=code,
            requirements="scanpy>=1.9.0\nscvelo>=0.2.5\nnumpy>=1.24.0",
            parameters={},
            static_config=config
        )
    
    def _create_velocity_prep_block(self):
        """Create velocity preparation block."""
        config = StaticConfig(
            args=[],
            description="Prepare data for velocity computation",
            tag="velocity_prep",
            source="mock"
        )
        
        code = '''
def run(adata, **kwargs):
    """Prepare data for velocity computation."""
    import scvelo as scv
    
    print("Preparing for velocity analysis...")
    
    # Filter and normalize for velocity
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    
    # Compute moments for velocity estimation
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    print("Data prepared for velocity computation")
    
    return adata
'''
        
        return NewFunctionBlock(
            name="velocity_preparation",
            type=FunctionBlockType.PYTHON,
            description="Prepare for velocity computation",
            code=code,
            requirements="scvelo>=0.2.5",
            parameters={},
            static_config=config
        )
    
    def _create_pseudotime_block(self):
        """Create pseudotime calculation block."""
        config = StaticConfig(
            args=[
                Arg(name="root_cluster", value_type="str", description="Root cluster for pseudotime",
                    optional=True, default_value=None)
            ],
            description="Calculate diffusion pseudotime",
            tag="pseudotime",
            source="mock"
        )
        
        code = '''
def run(adata, root_cluster=None, **kwargs):
    """Calculate diffusion pseudotime."""
    import scanpy as sc
    import numpy as np
    
    print("Calculating pseudotime...")
    
    # Diffusion map
    sc.tl.diffmap(adata)
    
    # Set root cell
    if root_cluster is not None:
        # Find cell in root cluster closest to origin
        root_mask = adata.obs['leiden'] == str(root_cluster)
        root_cells = np.where(root_mask)[0]
        if len(root_cells) > 0:
            adata.uns['iroot'] = root_cells[0]
    else:
        # Use cell with minimum DC1
        adata.uns['iroot'] = np.argmin(adata.obsm['X_diffmap'][:, 0])
    
    # Calculate pseudotime
    sc.tl.dpt(adata)
    
    print(f"Pseudotime calculated from root cell {adata.uns['iroot']}")
    
    return adata
'''
        
        return NewFunctionBlock(
            name="calculate_pseudotime",
            type=FunctionBlockType.PYTHON,
            description="Calculate diffusion pseudotime",
            code=code,
            requirements="scanpy>=1.9.0\nnumpy>=1.24.0",
            parameters={},
            static_config=config
        )
    
    def _create_scvelo_block(self):
        """Create scVelo analysis block."""
        config = StaticConfig(
            args=[
                Arg(name="mode", value_type="str", description="Velocity mode",
                    optional=True, default_value="dynamical")
            ],
            description="Run scVelo RNA velocity analysis",
            tag="velocity",
            source="mock"
        )
        
        code = '''
def run(adata, mode="dynamical", **kwargs):
    """Run scVelo RNA velocity analysis."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    
    print(f"Running scVelo in {mode} mode...")
    
    # Recover dynamics if using dynamical mode
    if mode == "dynamical":
        scv.tl.recover_dynamics(adata, n_jobs=4)
    
    # Compute velocity
    scv.tl.velocity(adata, mode=mode)
    
    # Compute velocity graph
    scv.tl.velocity_graph(adata)
    
    # Compute velocity confidence
    scv.tl.velocity_confidence(adata)
    
    # Compute velocity pseudotime
    scv.tl.velocity_pseudotime(adata)
    
    # Compute latent time if dynamical
    if mode == "dynamical":
        scv.tl.latent_time(adata)
    
    # Terminal states and lineages
    scv.tl.terminal_states(adata)
    
    print("Velocity analysis completed")
    
    # Save key velocity plot
    scv.pl.velocity_embedding_stream(adata, basis='umap', save='_velocity_stream.png', show=False)
    
    return adata
'''
        
        return NewFunctionBlock(
            name="scvelo_analysis",
            type=FunctionBlockType.PYTHON,
            description="Run scVelo RNA velocity",
            code=code,
            requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0",
            parameters={"mode": "dynamical"},
            static_config=config
        )
    
    def _create_velocity_plots_block(self):
        """Create velocity visualization block."""
        config = StaticConfig(
            args=[],
            description="Generate velocity visualizations",
            tag="velocity_plots",
            source="mock"
        )
        
        code = '''
def run(adata, **kwargs):
    """Generate velocity visualizations."""
    import scvelo as scv
    import matplotlib.pyplot as plt
    
    print("Generating velocity plots...")
    
    # Velocity embedding plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stream plot
    scv.pl.velocity_embedding_stream(adata, basis='umap', ax=axes[0, 0], show=False)
    axes[0, 0].set_title('Velocity Stream')
    
    # Grid plot
    scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120,
                              basis='umap', ax=axes[0, 1], show=False)
    axes[0, 1].set_title('Velocity Grid')
    
    # Velocity confidence
    scv.pl.scatter(adata, c='velocity_confidence', cmap='coolwarm',
                   basis='umap', ax=axes[1, 0], show=False)
    axes[1, 0].set_title('Velocity Confidence')
    
    # Velocity pseudotime
    scv.pl.scatter(adata, c='velocity_pseudotime', cmap='gnuplot',
                   basis='umap', ax=axes[1, 1], show=False)
    axes[1, 1].set_title('Velocity Pseudotime')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/velocity_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Phase portraits for top genes
    top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index[:6]
    scv.pl.velocity(adata, top_genes, ncols=3, save='_phase_portraits.png', show=False)
    
    # Heatmap
    scv.pl.heatmap(adata, adata.var['fit_likelihood'].sort_values(ascending=False).index[:100],
                   sortby='velocity_pseudotime', col_color='leiden',
                   save='_velocity_heatmap.png', show=False)
    
    print("Velocity plots generated")
    
    return adata
'''
        
        return NewFunctionBlock(
            name="velocity_visualizations",
            type=FunctionBlockType.PYTHON,
            description="Generate velocity plots",
            code=code,
            requirements="scvelo>=0.2.5\nmatplotlib>=3.6.0",
            parameters={},
            static_config=config
        )
    
    def _create_analysis_plots_block(self):
        """Create additional analysis plots block."""
        config = StaticConfig(
            args=[],
            description="Generate analysis summary plots",
            tag="analysis_plots",
            source="mock"
        )
        
        code = '''
def run(adata, **kwargs):
    """Generate analysis summary plots."""
    import scanpy as sc
    import scvelo as scv
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print("Generating analysis plots...")
    
    # Summary plot with multiple panels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # UMAP colored by clusters
    sc.pl.umap(adata, color='leiden', legend_loc='on data',
               ax=axes[0, 0], show=False, title='Clusters')
    
    # UMAP colored by pseudotime
    sc.pl.umap(adata, color='dpt_pseudotime', cmap='viridis',
               ax=axes[0, 1], show=False, title='Diffusion Pseudotime')
    
    # UMAP colored by latent time (if available)
    if 'latent_time' in adata.obs.columns:
        scv.pl.scatter(adata, c='latent_time', cmap='viridis',
                       basis='umap', ax=axes[0, 2], show=False, title='Latent Time')
    else:
        axes[0, 2].axis('off')
    
    # Terminal states
    if 'terminal_states_probs' in adata.obs.columns:
        scv.pl.scatter(adata, c='terminal_states_probs', cmap='viridis',
                       basis='umap', ax=axes[1, 0], show=False, title='Terminal States')
    
    # Gene expression for marker
    if 'highly_variable' in adata.var.columns:
        top_gene = adata.var[adata.var.highly_variable].index[0]
        sc.pl.umap(adata, color=top_gene, ax=axes[1, 1], show=False,
                   title=f'Expression: {top_gene}')
    
    # Velocity length
    scv.tl.velocity_graph(adata, compute_uncertainties=True)
    scv.pl.scatter(adata, c='velocity_length', cmap='coolwarm',
                   basis='umap', ax=axes[1, 2], show=False, title='Velocity Length')
    
    plt.tight_layout()
    plt.savefig('/workspace/output/figures/analysis_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save analysis summary
    summary = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_clusters': len(adata.obs['leiden'].unique()),
        'has_velocity': 'velocity_graph' in adata.uns,
        'velocity_genes': int(adata.var['velocity_genes'].sum()) if 'velocity_genes' in adata.var else 0
    }
    
    pd.DataFrame([summary]).to_csv('/workspace/output/analysis_summary.csv', index=False)
    
    print("Analysis complete!")
    
    return adata
'''
        
        return NewFunctionBlock(
            name="analysis_summary_plots",
            type=FunctionBlockType.PYTHON,
            description="Generate analysis summary",
            code=code,
            requirements="scanpy>=1.9.0\nscvelo>=0.2.5\nmatplotlib>=3.6.0\npandas>=2.0.0",
            parameters={},
            static_config=config
        )


def test_main_agent_scvelo_pipeline():
    """Test main agent with scvelo pipeline."""
    print("\n=== Testing Main Agent with scVelo Pipeline ===")
    
    # Setup paths
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    output_dir = Path("test_outputs/scvelo_agent")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_data.exists():
        print(f"Error: Test data not found at {input_data}")
        sys.exit(1)
    
    # Create components
    tree_manager = AnalysisTreeManager()
    mock_selector = MockFunctionSelector()
    bug_fixer = BugFixerAgent()  # No LLM service
    orchestrator = OrchestratorAgent(tree_manager, mock_selector, bug_fixer)
    executor_manager = ExecutorManager()
    node_executor = NodeExecutor(executor_manager)
    
    # Create main agent with refactored components
    class RefactoredMainAgent(MainAgent):
        """Main agent using refactored components."""
        
        def __init__(self):
            # Don't call super().__init__ to avoid OpenAI requirement
            self.executor_manager = executor_manager
            self.tree_manager = tree_manager
            self.node_executor = node_executor
            self.orchestrator = orchestrator
            self.function_selector = mock_selector
            self.bug_fixer = bug_fixer
            
        def run_refactored_analysis(
            self,
            input_data_path: str,
            user_request: str,
            output_dir: Path,
            max_nodes: int = 20
        ):
            """Run analysis using refactored agent architecture."""
            
            # Create analysis tree
            tree = tree_manager.create_tree(
                user_request=user_request,
                input_data_path=str(input_data_path),
                max_nodes=max_nodes,
                max_children_per_node=3,
                max_debug_trials=3,
                generation_mode=GenerationMode.ONLY_NEW
            )
            
            print(f"\nUser Request: {user_request}")
            print(f"Max Nodes: {max_nodes}")
            
            # Main execution loop
            iteration = 0
            max_iterations = 10
            current_data_path = Path(input_data_path)
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                # Let orchestrator handle the workflow
                result = orchestrator.process({
                    'tree': tree,
                    'user_request': user_request,
                    'max_iterations': 1,  # One iteration at a time
                    'max_children': 3,
                    'input_data_path': str(current_data_path)
                })
                
                # Execute pending nodes
                pending_nodes = tree_manager.get_execution_order()
                
                for node in pending_nodes:
                    print(f"\nExecuting: {node.function_block.name}")
                    
                    # Update state
                    tree_manager.update_node_execution(node.id, 'running')
                    
                    # Get input data path
                    node_input_path = tree_manager.get_latest_data_path(node.id) or current_data_path
                    
                    # Execute node
                    state, exec_result = node_executor.execute_node(node=node, tree=tree, input_path=node_input_path, output_base_dir=output_dir / tree.id
                    )
                    
                    # Handle result
                    if state == 'completed':
                        tree_manager.update_node_execution(
                            node.id,
                            state=state,
                            output_data_id=exec_result.output_data_path,
                            figures=exec_result.figures,
                            logs=[exec_result.logs] if exec_result.logs else [],
                            duration=exec_result.duration
                        )
                        
                        if exec_result.output_data_path:
                            current_data_path = Path(exec_result.output_data_path)
                            
                        print(f"✓ Completed in {exec_result.duration:.1f}s")
                        if exec_result.figures:
                            print(f"  Generated {len(exec_result.figures)} figures")
                            
                    elif state == 'failed':
                        # Try to fix
                        if orchestrator.handle_failed_node(
                            node, 
                            exec_result.error or "Unknown error",
                            exec_result.stdout,
                            exec_result.stderr
                        ):
                            print(f"! Attempting to fix error...")
                        else:
                            tree_manager.update_node_execution(
                                node.id,
                                state=state,
                                error=exec_result.error,
                                logs=[exec_result.logs] if exec_result.logs else []
                            )
                            print(f"✗ Failed: {exec_result.error}")
                
                # Save tree state
                tree_path = output_dir / "analysis_tree.json"
                tree_manager.save_tree(tree_path)
                
                # Check if satisfied
                if result['satisfied']:
                    print("\n✓ Analysis request satisfied!")
                    break
                    
                # Check if we can continue
                if not tree_manager.can_continue_expansion():
                    print("\n! Cannot continue expansion")
                    break
            
            # Display final summary
            summary = tree_manager.get_summary()
            print("\n=== Analysis Summary ===")
            print(f"Total Nodes: {summary['total_nodes']}")
            print(f"Completed: {summary['completed_nodes']}")
            print(f"Failed: {summary['failed_nodes']}")
            print(f"Total Duration: {summary.get('total_duration_seconds', 0):.1f}s")
            
            # Display tree structure
            print("\n=== Analysis Tree ===")
            self._print_tree(tree)
            
            return summary
            
        def _print_tree(self, tree, node_id=None, prefix="", is_last=True):
            """Print tree structure."""
            if node_id is None:
                if tree.root_node_id:
                    self._print_tree(tree, tree.root_node_id)
                return
                
            node = tree.get_node(node_id)
            if not node:
                return
                
            # Print current node
            connector = "└── " if is_last else "├── "
            status = {
                'completed': '✓',
                'failed': '✗',
                'pending': '○',
                'running': '●'
            }.get(node.state, '?')
            
            print(f"{prefix}{connector}{status} {node.function_block.name}")
            
            # Print children
            children = node.children
            for i, child_id in enumerate(children):
                is_last_child = i == len(children) - 1
                extension = "    " if is_last else "│   "
                self._print_tree(tree, child_id, prefix + extension, is_last_child)
    
    # Create and run agent
    agent = RefactoredMainAgent()
    
    # Validate environment
    print("\nValidating environment...")
    validation = executor_manager.validate_environment()
    for check, result in validation.items():
        print(f"  {'✓' if result else '✗'} {check}")
    
    if not validation.get('docker_available'):
        print("Error: Docker is not available")
        sys.exit(1)
    
    # Run analysis
    user_request = "Analyze the scRNA-seq data with RNA velocity analysis using scVelo. Calculate pseudotime and generate comprehensive visualizations."
    
    summary = agent.run_refactored_analysis(
        input_data_path=str(input_data),
        user_request=user_request,
        output_dir=output_dir,
        max_nodes=20
    )
    
    # List output files
    print("\n=== Output Files ===")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and not path.name.startswith('.'):
            print(f"  {path.relative_to(output_dir)}")
    
    print("\n=== Test Complete ===")
    return summary


if __name__ == "__main__":
    test_main_agent_scvelo_pipeline()