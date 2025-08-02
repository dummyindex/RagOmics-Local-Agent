"""Main agent that coordinates the entire analysis workflow."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

from ..models import (
    NodeState, GenerationMode, AnalysisTree, NewFunctionBlock,
    FunctionBlockType, StaticConfig, Arg
)
from ..analysis_tree_management import AnalysisTreeManager, NodeExecutor
from ..job_executors import ExecutorManager
from ..llm_service import OpenAIService
from ..utils import setup_logger
from .orchestrator_agent import OrchestratorAgent
from .function_selector_agent import FunctionSelectorAgent
from .function_creator_agent import FunctionCreatorAgent
from .bug_fixer_agent import BugFixerAgent

logger = setup_logger(__name__)


class MainAgent:
    """Main agent that provides the high-level interface for analysis."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the main agent.
        
        Args:
            openai_api_key: OpenAI API key. If not provided, will try to read from environment.
        """
        # Get API key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Try to read from .env file
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.startswith("OPENAI_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip()
                            break
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Some features may not work.")
        
        # Initialize components
        self.tree_manager = AnalysisTreeManager()
        self.executor_manager = ExecutorManager()
        self.node_executor = NodeExecutor(self.executor_manager)
        
        # Initialize LLM service if API key is available
        if self.api_key:
            self.llm_service = OpenAIService(api_key=self.api_key)
            
            # Initialize agents
            self.function_creator = FunctionCreatorAgent(llm_service=self.llm_service)
            self.function_selector = FunctionSelectorAgent(
                llm_service=self.llm_service,
                function_creator=self.function_creator
            )
            self.bug_fixer = BugFixerAgent(llm_service=self.llm_service)
            self.orchestrator = OrchestratorAgent(
                tree_manager=self.tree_manager,
                function_selector=self.function_selector,
                bug_fixer=self.bug_fixer
            )
        else:
            self.llm_service = None
            self.function_creator = None
            self.function_selector = None
            self.bug_fixer = None
            self.orchestrator = None
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate the execution environment.
        
        Returns:
            Dictionary with validation results for each component.
        """
        return self.executor_manager.validate_environment()
    
    def run_analysis(
        self,
        input_data_path: Union[str, Path],
        user_request: str,
        output_dir: Optional[Union[str, Path]] = None,
        max_nodes: int = 20,
        max_children: int = 3,
        max_debug_trials: int = 3,
        generation_mode: str = "mixed",
        llm_model: str = "gpt-4o-mini",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run the complete analysis workflow.
        
        Args:
            input_data_path: Path to input data file
            user_request: Natural language request from user
            output_dir: Output directory (will be created if doesn't exist)
            max_nodes: Maximum number of analysis nodes
            max_children: Maximum children per node
            max_debug_trials: Maximum debug attempts per node
            generation_mode: "mixed", "only_new", or "only_existing"
            llm_model: OpenAI model to use
            verbose: Whether to print verbose output
            
        Returns:
            Dictionary with results including output directory and execution statistics
        """
        # Setup paths
        input_path = Path(input_data_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_dir is None:
            output_dir = Path("outputs") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main agent task folder with timestamp
        task_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_id = str(uuid.uuid4())[:8]
        main_task_id = f"main_{task_timestamp}_{temp_id}"
        main_task_dir = output_dir / main_task_id
        main_task_dir.mkdir(parents=True, exist_ok=True)
        
        # Create orchestrator subfolder
        orchestrator_dir = main_task_dir / f"orchestrator_{task_timestamp}"
        orchestrator_dir.mkdir(parents=True, exist_ok=True)
        
        # Store main task metadata
        main_task_metadata = {
            'task_id': main_task_id,
            'user_request': user_request,
            'input_data_path': str(input_path),
            'created_at': datetime.now().isoformat(),
            'llm_model': llm_model,
            'generation_mode': generation_mode
        }
        
        with open(main_task_dir / 'task_metadata.json', 'w') as f:
            json.dump(main_task_metadata, f, indent=2)
        
        # Set model for LLM service
        if self.llm_service and llm_model:
            self.llm_service.model = llm_model
        
        # Create analysis tree
        logger.info(f"Creating analysis tree for request: {user_request}")
        tree = self.tree_manager.create_tree(
            user_request=user_request,
            input_data_path=str(input_path),
            max_nodes=max_nodes,
            max_children_per_node=max_children,
            max_debug_trials=max_debug_trials,
            generation_mode=GenerationMode(generation_mode),
            llm_model=llm_model
        )
        
        # Update main task ID with actual tree ID
        updated_main_task_id = f"main_{task_timestamp}_{tree.id[:8]}"
        updated_main_task_dir = output_dir / updated_main_task_id
        if main_task_dir.name != updated_main_task_dir.name:
            main_task_dir.rename(updated_main_task_dir)
            main_task_dir = updated_main_task_dir
            orchestrator_dir = main_task_dir / f"orchestrator_{task_timestamp}"
        
        # Create tree directory with new structure
        tree_dir = output_dir / f"tree_{tree.id}"
        tree_dir.mkdir(parents=True, exist_ok=True)
        
        # Create nodes directory
        nodes_dir = tree_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tree-level agent_tasks directory
        tree_agent_tasks_dir = tree_dir / "agent_tasks"
        tree_agent_tasks_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tree metadata
        tree_metadata = {
            "id": tree.id,
            "user_request": tree.user_request,
            "input_data_path": tree.input_data_path,
            "created_at": datetime.now().isoformat(),
            "generation_mode": tree.generation_mode.value if hasattr(tree.generation_mode, 'value') else str(tree.generation_mode),
            "max_nodes": tree.max_nodes,
            "max_children_per_node": tree.max_children_per_node,
            "max_debug_trials": tree.max_debug_trials
        }
        with open(tree_dir / "tree_metadata.json", 'w') as f:
            json.dump(tree_metadata, f, indent=2)
        
        # Save analysis tree
        tree_file = tree_dir / "analysis_tree.json"
        self.tree_manager.save_tree(tree_file)
        
        # Also save in orchestrator folder for reference
        self.tree_manager.save_tree(orchestrator_dir / "analysis_tree.json")
        
        if not self.llm_service:
            # If no LLM service, create a simple preprocessing node for testing
            logger.warning("No LLM service available. Creating default preprocessing node.")
            self._create_default_node(tree, input_path)
        else:
            # Use LLM to generate analysis plan
            self._generate_analysis_plan(tree, user_request, input_path, output_dir, verbose, orchestrator_dir)
        
        # Execute the analysis tree
        results = self._execute_tree(tree, input_path, output_dir, verbose, main_task_dir)
        
        # Save final tree in tree directory
        tree_dir = output_dir / f"tree_{tree.id}"
        tree_file = tree_dir / "analysis_tree.json"
        self.tree_manager.save_tree(tree_file)
        
        # Prepare result summary with new structure paths
        return {
            "output_dir": str(output_dir),
            "tree_dir": str(tree_dir),
            "tree_id": tree.id,
            "total_nodes": tree.total_nodes,
            "completed_nodes": tree.completed_nodes,
            "failed_nodes": tree.failed_nodes,
            "tree_file": str(tree_file),
            "nodes_dir": str(tree_dir / "nodes"),
            "results": results
        }
    
    def _create_default_node(self, tree: AnalysisTree, input_path: Path):
        """Create a default preprocessing node for testing."""
        # Create a simple preprocessing block
        code = '''
def run(adata, **parameters):
    """Basic preprocessing for single-cell data."""
    import scanpy as sc
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Filter cells
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    
    # PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # UMAP
    sc.tl.umap(adata)
    
    # Clustering
    sc.tl.leiden(adata)
    
    print(f"Preprocessed data: {adata.shape}")
    return adata
'''
        
        static_config = StaticConfig(
            args=[],
            description="Basic preprocessing for single-cell data",
            tag="preprocessing"
        )
        
        block = NewFunctionBlock(
            name="basic_preprocessing",
            type=FunctionBlockType.PYTHON,
            description="Basic preprocessing pipeline",
            static_config=static_config,
            code=code,
            requirements="scanpy\nnumpy\npandas",
            parameters={}
        )
        
        # Add as root node
        self.tree_manager.add_root_node(block)
    
    def _generate_analysis_plan(
        self, 
        tree: AnalysisTree, 
        user_request: str,
        input_path: Path,
        output_dir: Path,
        verbose: bool,
        orchestrator_dir: Path
    ):
        """Generate analysis plan using LLM."""
        logger.info("Generating analysis plan with LLM...")
        
        # Use function selector to generate initial function blocks
        context = {
            "user_request": user_request,
            "input_data_path": str(input_path),
            "tree_state": self._get_tree_state(tree),
            "generation_mode": tree.generation_mode,
            "tree": tree,
            "max_children": tree.max_children_per_node
        }
        
        try:
            # Track orchestrator task for function selection
            selector_task_id = f"selector_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}"
            selector_task_dir = orchestrator_dir / selector_task_id
            selector_task_dir.mkdir(parents=True, exist_ok=True)
            
            # Save context for selector
            with open(selector_task_dir / 'context.json', 'w') as f:
                json.dump({
                    'user_request': user_request,
                    'input_data_path': str(input_path),
                    'generation_mode': tree.generation_mode.value if hasattr(tree.generation_mode, 'value') else str(tree.generation_mode),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            # Get function block recommendations
            recommendations = self.function_selector.process(context)
            
            # Save selector results
            with open(selector_task_dir / 'results.json', 'w') as f:
                json.dump({
                    'satisfied': recommendations.get('satisfied', False),
                    'reasoning': recommendations.get('reasoning', ''),
                    'num_blocks': len(recommendations.get('function_blocks', [])),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            function_blocks = recommendations.get('function_blocks', [])
            
            if verbose:
                logger.info(f"Generated {len(function_blocks)} function blocks")
            
            # Add function blocks to tree and track creation tasks
            for i, block in enumerate(function_blocks):
                if i >= tree.max_nodes:
                    break
                
                # The block is already a NewFunctionBlock or ExistingFunctionBlock object
                if i == 0:
                    # Add as root node
                    node = self.tree_manager.add_root_node(block)
                else:
                    # Add as child of previous node
                    parent_id = list(tree.nodes.keys())[i-1]
                    nodes = self.tree_manager.add_child_nodes(parent_id, [block])
                    if nodes:
                        node = nodes[0]
                
                # Track function creation in node's directory with new structure
                if node:
                    # Create node directory in the new structure
                    tree_dir = output_dir / f"tree_{tree.id}"
                    node_dir = tree_dir / "nodes" / f"node_{node.id}"
                    node_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create agent_tasks directory for this node
                    agent_tasks_dir = node_dir / "agent_tasks"
                    agent_tasks_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save node info
                    node_info = {
                        "id": node.id,
                        "name": block.name,
                        "type": block.type.value if hasattr(block.type, 'value') else str(block.type),
                        "parent_id": node.parent_id,
                        "children_ids": node.children,
                        "state": "pending",
                        "created_at": datetime.now().isoformat(),
                        "level": node.level
                    }
                    with open(node_dir / "node_info.json", 'w') as f:
                        json.dump(node_info, f, indent=2)
                    
                    # Create creator task directory in node's agent_tasks
                    creator_task_id = f"creator_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}_{node.id[:8]}"
                    creator_task_dir = agent_tasks_dir / creator_task_id
                    creator_task_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save creator task info
                    creator_task_record = {
                        'task_id': creator_task_id,
                        'node_id': node.id,
                        'function_block_name': block.name,
                        'function_block_type': block.type.value if hasattr(block.type, 'value') else str(block.type),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save in node's agent_tasks folder
                    with open(creator_task_dir / 'task_info.json', 'w') as f:
                        json.dump(creator_task_record, f, indent=2)
                    
                    # Save the generated code
                    if hasattr(block, 'code'):
                        with open(creator_task_dir / 'generated_code.py', 'w') as f:
                            f.write(block.code)
                    
                    # Also save reference in orchestrator folder
                    with open(orchestrator_dir / f'{creator_task_id}.json', 'w') as f:
                        json.dump({
                            'task_id': creator_task_id,
                            'node_id': node.id,
                            'function_block_name': block.name,
                            'task_location': str(creator_task_dir),
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                
                if verbose:
                    logger.info(f"Added node: {block.name}")
                    
        except Exception as e:
            logger.error(f"Error generating analysis plan: {e}")
            # Fall back to default node
            self._create_default_node(tree, input_path)
    
    
    def _execute_tree(
        self, 
        tree: AnalysisTree,
        input_path: Path,
        output_dir: Path,
        verbose: bool,
        main_task_dir: Path
    ) -> Dict[str, Any]:
        """Execute all nodes in the analysis tree."""
        results = {}
        
        # Execute nodes in order (by level)
        nodes_by_level = {}
        for node in tree.nodes.values():
            level = node.level
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)
        
        current_input = input_path
        
        for level in sorted(nodes_by_level.keys()):
            for node in nodes_by_level[level]:
                if node.state == NodeState.COMPLETED:
                    continue
                
                logger.info(f"Executing node: {node.function_block.name}")
                
                # Update state
                self.tree_manager.update_node_execution(node.id, NodeState.RUNNING)
                
                try:
                    # Use new structure: tree_TREEID/nodes/node_NODEID
                    tree_dir = output_dir / f"tree_{tree.id}"
                    node_dir = tree_dir / "nodes" / f"node_{node.id}"
                    
                    # Node directory should already exist from generation phase
                    if not node_dir.exists():
                        node_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Create node info if it doesn't exist
                        node_info = {
                            "id": node.id,
                            "name": node.function_block.name,
                            "type": node.function_block.type.value if hasattr(node.function_block.type, 'value') else str(node.function_block.type),
                            "parent_id": node.parent_id,
                            "children_ids": node.children,
                            "state": "running",
                            "created_at": datetime.now().isoformat(),
                            "level": node.level
                        }
                        with open(node_dir / "node_info.json", 'w') as f:
                            json.dump(node_info, f, indent=2)
                    
                    # Get agent_tasks directory (should exist)
                    agent_tasks_dir = node_dir / "agent_tasks"
                    if not agent_tasks_dir.exists():
                        agent_tasks_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Execute node (this will create job directories under node_dir)
                    state, output_path = self.node_executor.execute_node(
                        node=node,
                        tree=tree,
                        input_path=current_input,
                        output_base_dir=output_dir
                    )
                    
                    # Update tree
                    self.tree_manager.update_node_execution(
                        node.id, 
                        state,
                        output_data_id=output_path
                    )
                    
                    if state == NodeState.COMPLETED and output_path:
                        # With new structure, output_path points to outputs directory
                        current_input = Path(output_path) / "output_data.h5ad"
                        if not current_input.exists():
                            # Try alternative path
                            current_input = Path(output_path)
                        
                        results[node.id] = {
                            "name": node.function_block.name,
                            "state": "completed",
                            "output": str(output_path)
                        }
                        
                        if verbose:
                            logger.info(f"  ✓ Completed: {node.function_block.name}")
                    else:
                        # Try to fix with bug fixer if available
                        if self.bug_fixer and node.debug_attempts < tree.max_debug_trials:
                            logger.info(f"Attempting to fix failed node: {node.function_block.name}")
                            
                            # Create bug fixer task directory
                            fixer_task_id = f"fixer_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}_{node.id[:8]}"
                            fixer_task_dir = agent_tasks_dir / fixer_task_id
                            fixer_task_dir.mkdir(parents=True, exist_ok=True)
                            
                            fixed_block = self._fix_failed_node(node, tree, fixer_task_dir)
                            if fixed_block:
                                # Update node with fixed block
                                node.function_block = fixed_block
                                node.debug_attempts += 1
                                # Retry execution
                                state, output_path = self.node_executor.execute_node(
                                    node=node,
                                    tree=tree,
                                    input_path=current_input,
                                    output_base_dir=output_dir
                                )
                                
                                if state == NodeState.COMPLETED:
                                    self.tree_manager.update_node_execution(
                                        node.id,
                                        state,
                                        output_data_id=output_path
                                    )
                                    # Update input path for next node
                                current_input = Path(output_path) / "output_data.h5ad"
                                if not current_input.exists():
                                    current_input = Path(output_path)
                                    results[node.id] = {
                                        "name": node.function_block.name,
                                        "state": "completed_after_fix",
                                        "output": str(output_path)
                                    }
                        
                        if node.state != NodeState.COMPLETED:
                            results[node.id] = {
                                "name": node.function_block.name,
                                "state": "failed",
                                "error": node.error
                            }
                            
                            if verbose:
                                logger.error(f"  ✗ Failed: {node.function_block.name}")
                                
                except Exception as e:
                    logger.error(f"Error executing node {node.id}: {e}")
                    self.tree_manager.update_node_execution(
                        node.id,
                        NodeState.FAILED,
                        error=str(e)
                    )
                    results[node.id] = {
                        "name": node.function_block.name,
                        "state": "failed",
                        "error": str(e)
                    }
        
        return results
    
    def _fix_failed_node(self, node: Any, tree: AnalysisTree, fixer_task_dir: Optional[Path] = None) -> Optional[NewFunctionBlock]:
        """Attempt to fix a failed node using bug fixer agent."""
        if not self.bug_fixer:
            return None
            
        # Parse error and logs to get stdout/stderr
        error_message = node.error or ""
        stdout = ""
        stderr = ""
        
        # Extract from logs if available
        if isinstance(node.logs, list):
            stdout = "\n".join([log for log in node.logs if not log.startswith("ERROR")])
            stderr = "\n".join([log for log in node.logs if log.startswith("ERROR")])
        elif isinstance(node.logs, str):
            stdout = node.logs
        
        # If error contains stderr, extract it
        if "STDERR:" in error_message:
            parts = error_message.split("STDERR:")
            stderr = parts[1] if len(parts) > 1 else stderr
            error_message = parts[0]
        
        context = {
            "function_block": node.function_block,
            "error_message": error_message,
            "stdout": stdout,
            "stderr": stderr
        }
        
        # Save bug fixer context if task dir provided
        if fixer_task_dir:
            with open(fixer_task_dir / 'context.json', 'w') as f:
                json.dump({
                    'node_id': node.id,
                    'function_block_name': node.function_block.name,
                    'error_message': error_message[:1000],  # Truncate for storage
                    'debug_attempt': node.debug_attempts + 1,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        
        try:
            result = self.bug_fixer.process(context)
            if result and result.get('success') and result.get('fixed_code'):
                # Create a new function block with the fixed code
                fixed_block = NewFunctionBlock(
                    name=node.function_block.name,
                    type=node.function_block.type,
                    description=node.function_block.description,
                    code=result['fixed_code'],
                    requirements=result.get('fixed_requirements', node.function_block.requirements),
                    parameters=node.function_block.parameters,
                    static_config=node.function_block.static_config
                )
                
                # Save fix result if task dir provided
                if fixer_task_dir:
                    with open(fixer_task_dir / 'result.json', 'w') as f:
                        json.dump({
                            'success': True,
                            'reasoning': result.get('reasoning', 'Fixed using bug fixer'),
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                    
                    # Save fixed code
                    with open(fixer_task_dir / 'fixed_code.py', 'w') as f:
                        f.write(result['fixed_code'])
                
                logger.info(f"Successfully generated fix for {node.function_block.name}")
                return fixed_block
        except Exception as e:
            logger.error(f"Error fixing node: {e}")
            
        return None
    
    def _get_tree_state(self, tree: AnalysisTree) -> Dict[str, Any]:
        """Get current state of the analysis tree."""
        return {
            "total_nodes": tree.total_nodes,
            "completed_nodes": tree.completed_nodes,
            "failed_nodes": tree.failed_nodes,
            "nodes": [
                {
                    "id": node.id,
                    "name": node.function_block.name,
                    "state": node.state,
                    "level": node.level
                }
                for node in tree.nodes.values()
            ]
        }