"""Main agent that coordinates the entire analysis workflow."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

from ..models import (
    NodeState, GenerationMode, AnalysisTree, AnalysisNode,
    NewFunctionBlock, ExistingFunctionBlock,
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
    """Main agent that coordinates the entire analysis workflow."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the main agent.
        
        Args:
            openai_api_key: OpenAI API key for LLM services
        """
        # Initialize LLM service
        self.llm_service = OpenAIService(api_key=openai_api_key) if openai_api_key else None
        
        # Initialize managers
        self.tree_manager = AnalysisTreeManager()
        self.executor_manager = ExecutorManager()
        self.node_executor = NodeExecutor(self.executor_manager)
        
        # Initialize specialized agents
        self.orchestrator = OrchestratorAgent(self.llm_service) if self.llm_service else None
        self.function_selector = FunctionSelectorAgent(self.llm_service) if self.llm_service else None
        self.function_creator = FunctionCreatorAgent(self.llm_service) if self.llm_service else None
        self.bug_fixer = BugFixerAgent(self.llm_service) if self.llm_service else None
        
        logger.info("Main Agent initialized")
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate the execution environment.
        
        Returns:
            Dictionary with validation results for each component
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
        max_iterations: int = 10,  # Add safeguard parameter
        generation_mode: str = "mixed",
        llm_model: str = "gpt-4o-mini",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run the complete analysis workflow.
        
        Args:
            input_data_path: Path to input data (file or directory)
            user_request: Natural language request from user
            output_dir: Output directory
            max_nodes: Maximum number of analysis nodes
            max_children: Maximum children per node
            max_debug_trials: Maximum debug attempts per node
            max_iterations: Maximum planning iterations (safeguard against infinite loops)
            generation_mode: "mixed", "only_new", or "only_existing"
            llm_model: OpenAI model to use
            verbose: Whether to print verbose output
            
        Returns:
            Dictionary with results including output directory and execution statistics
        """
        # Validate input path
        input_path = Path(input_data_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path("outputs") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Create main agent task directory
        main_task_dir = output_dir / f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tree.id[:8]}"
        main_task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save user request
        with open(main_task_dir / "user_request.txt", 'w') as f:
            f.write(user_request)
        
        # Save agent info
        agent_info = {
            "agent_type": "main",
            "tree_id": tree.id,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "created_at": datetime.now().isoformat(),
            "llm_model": llm_model,
            "generation_mode": generation_mode
        }
        with open(main_task_dir / "agent_info.json", 'w') as f:
            json.dump(agent_info, f, indent=2)
        
        results = {}  # Initialize results
        try:
            # Generate and execute analysis plan
            if self.orchestrator:
                self._generate_and_execute_plan(
                    tree, input_path, output_dir, verbose, main_task_dir,
                    max_iterations=max_iterations
                )
            else:
                # Fallback for testing without LLM
                self._create_default_pipeline(tree)
            
            # Execute the tree
            results = self._execute_tree(tree, input_path, output_dir, verbose)
            
        finally:
            # Always save tree state, even if execution was interrupted
            tree_file = output_dir / "analysis_tree.json"
            self.tree_manager.save_tree(tree_file)
            
            # Also save in tree directory
            tree_dir = output_dir / tree.id
            if tree_dir.exists():
                tree_file_in_tree = tree_dir / "analysis_tree.json"
                self.tree_manager.save_tree(tree_file_in_tree)
            
            # Create directory tree markdown
            self.tree_manager._create_directory_tree_md(tree, output_dir)
        
        # Return summary
        return {
            "tree_id": tree.id,
            "output_dir": str(output_dir),
            "tree_file": str(tree_file),
            "total_nodes": tree.total_nodes,
            "completed_nodes": tree.completed_nodes,
            "failed_nodes": tree.failed_nodes,
            "results": results
        }
    
    def _generate_and_execute_plan(
        self,
        tree: AnalysisTree,
        input_path: Path,
        output_dir: Path,
        verbose: bool,
        main_task_dir: Path,
        max_iterations: int = 10  # Add safeguard
    ):
        """Generate analysis plan using orchestrator and execute it.
        
        This method implements the correct iterative tree expansion logic:
        1. Create root node if tree is empty
        2. Execute pending nodes
        3. For each successful node, ask LLM to create children
        4. Only expand tree from successful nodes
        5. Repeat until user request is satisfied or limits reached
        """
        orchestrator_dir = main_task_dir / "orchestrator_tasks"
        orchestrator_dir.mkdir(exist_ok=True)
        
        # Track execution state
        iteration = 0
        satisfied = False
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while not satisfied and tree.total_nodes < tree.max_nodes and iteration < max_iterations:
            iteration += 1
            
            if verbose:
                logger.info(f"Planning iteration {iteration} (max: {max_iterations})")
            
            # First iteration: Create root node
            if tree.total_nodes == 0:
                # Get initial recommendation for root node
                orchestrator_task = {
                    "user_request": tree.user_request,
                    "tree_state": self._get_tree_state(tree),
                    "iteration": iteration,
                    "max_nodes_remaining": tree.max_nodes - tree.total_nodes,
                    "phase": "root"
                }
                
                recommendations = self.orchestrator.plan_next_steps(orchestrator_task)
                
                # Create root node
                try:
                    function_blocks = self._process_recommendations(
                        recommendations, tree, orchestrator_dir, output_dir
                    )
                    
                    if function_blocks:
                        # Add only the first block as root
                        block = function_blocks[0]
                        node = self.tree_manager.add_root_node(block)
                        
                        if node:
                            self._log_function_creation_to_node(
                                node=node,
                                tree=tree,
                                output_dir=output_dir,
                                block=block,
                                iteration=iteration,
                                block_index=0
                            )
                            
                            if verbose:
                                logger.info(f"Added root node: {node.function_block.name}")
                            
                            # Execute the root node immediately
                            success = self._execute_single_node(node, tree, input_path, output_dir, verbose)
                            
                            if not success:
                                logger.error(f"Root node failed, stopping tree expansion")
                                break
                except Exception as e:
                    logger.error(f"Failed to create root node: {e}")
                    break
            
            else:
                # Subsequent iterations: Expand from successful nodes only
                successful_leaf_nodes = [
                    (node_id, node) for node_id, node in tree.nodes.items()
                    if node.state == NodeState.COMPLETED and len(node.children) == 0
                ]
                
                if not successful_leaf_nodes:
                    # No successful leaf nodes to expand from
                    if verbose:
                        logger.info("No successful leaf nodes to expand")
                    break
                
                # For each successful leaf node, ask LLM if we should create children
                for parent_id, parent_node in successful_leaf_nodes:
                    if tree.total_nodes >= tree.max_nodes:
                        break
                    
                    # Check if user request is satisfied
                    orchestrator_task = {
                        "user_request": tree.user_request,
                        "tree_state": self._get_tree_state(tree),
                        "iteration": iteration,
                        "parent_node": {
                            "id": parent_id,
                            "name": parent_node.function_block.name,
                            "output": parent_node.output_data_id
                        },
                        "phase": "expansion"
                    }
                    
                    recommendations = self.orchestrator.plan_next_steps(orchestrator_task)
                    satisfied = recommendations.get("satisfied", False)
                    
                    if satisfied:
                        if verbose:
                            logger.info("User request satisfied")
                        break
                    
                    # Process recommendations for this parent
                    try:
                        function_blocks = self._process_recommendations(
                            recommendations, tree, orchestrator_dir, output_dir, parent_id
                        )
                        
                        # Add children to this successful parent
                        for i, block in enumerate(function_blocks):
                            if tree.total_nodes >= tree.max_nodes:
                                break
                            
                            nodes = self.tree_manager.add_child_nodes(parent_id, [block])
                            if nodes:
                                child_node = nodes[0]
                                
                                self._log_function_creation_to_node(
                                    node=child_node,
                                    tree=tree,
                                    output_dir=output_dir,
                                    block=block,
                                    iteration=iteration,
                                    block_index=i
                                )
                                
                                if verbose:
                                    logger.info(f"Added child node: {child_node.function_block.name} to parent {parent_node.function_block.name}")
                                
                                # Execute the child node immediately
                                success = self._execute_single_node(child_node, tree, input_path, output_dir, verbose)
                                
                                if not success:
                                    logger.warning(f"Child node {child_node.function_block.name} failed")
                                    # Continue with other children or parents
                                
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(f"Failed to process children for {parent_node.function_block.name}: {e}")
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("Too many consecutive failures")
                            break
        
        if iteration >= max_iterations:
            logger.warning(f"Reached maximum iterations limit: {max_iterations}")
        elif tree.total_nodes >= tree.max_nodes:
            logger.warning(f"Reached maximum nodes limit: {tree.max_nodes}")
    
    def _execute_single_node(
        self,
        node: AnalysisNode,
        tree: AnalysisTree,
        input_path: Path,
        output_dir: Path,
        verbose: bool
    ) -> bool:
        """Execute a single node and return success status.
        
        Args:
            node: Node to execute
            tree: Analysis tree
            input_path: Input data path for root, or parent output for children
            output_dir: Base output directory
            verbose: Whether to log details
            
        Returns:
            True if node executed successfully, False otherwise
        """
        try:
            # Determine input for this node
            if node.parent_id:
                parent = tree.nodes.get(node.parent_id)
                if parent and parent.output_data_id:
                    node_input = Path(parent.output_data_id)
                else:
                    logger.error(f"Parent node {node.parent_id} has no output")
                    return False
            else:
                node_input = input_path
            
            # Execute the node
            if verbose:
                logger.info(f"Executing node: {node.function_block.name}")
            
            state, output_path = self.node_executor.execute_node(
                node=node,
                tree=tree,
                input_path=node_input,
                output_base_dir=output_dir
            )
            
            # Update tree state
            self.tree_manager.update_node_state(node.id, state, output_data_id=output_path)
            
            # Handle success
            if state == NodeState.COMPLETED:
                if verbose:
                    logger.info(f"  ✓ Node completed successfully: {node.function_block.name}")
                return True
            
            # Handle failure - attempt to fix
            if state == NodeState.FAILED:
                logger.error(f"  ✗ Node failed: {node.function_block.name}")
                
                if self.bug_fixer and node.debug_attempts < tree.max_debug_trials:
                    if verbose:
                        logger.info(f"Attempting to fix {node.function_block.name}")
                    
                    # Try to fix the node
                    self._attempt_fix(node, tree, node_input, output_dir, verbose)
                    
                    # Check if fix succeeded
                    if node.state == NodeState.COMPLETED:
                        return True
                
                return False
            
            # Pending or other states
            return False
            
        except Exception as e:
            logger.error(f"Error executing node {node.function_block.name}: {e}")
            self.tree_manager.update_node_state(node.id, NodeState.FAILED, error=str(e))
            return False
    
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
                    "state": node.state.value,
                    "level": node.level,
                    "parent_id": node.parent_id,
                    "children": node.children
                }
                for node in tree.nodes.values()
            ]
        }
    
    def _process_recommendations(
        self,
        recommendations: Dict[str, Any],
        tree: AnalysisTree,
        orchestrator_dir: Path,
        output_dir: Path = None,
        parent_node_id: str = None
    ) -> List[Union[NewFunctionBlock, ExistingFunctionBlock]]:
        """Process orchestrator recommendations to create/select function blocks."""
        function_blocks = []
        
        # Get recommended actions
        actions = recommendations.get("next_actions", [])
        
        # Get parent node's data structure if available
        parent_context = None
        if parent_node_id and output_dir:
            parent_node = tree.nodes.get(parent_node_id)
            if parent_node and parent_node.output_data_id:
                # Try to read parent's data structure
                parent_output_dir = Path(parent_node.output_data_id).parent
                data_structure_file = parent_output_dir / "_data_structure.json"
                if data_structure_file.exists():
                    try:
                        with open(data_structure_file, 'r') as f:
                            parent_context = json.load(f)
                    except Exception as e:
                        logger.debug(f"Could not read parent data structure: {e}")
        
        for action in actions:
            if action["type"] == "create_new":
                # Use function creator to generate new block
                if self.function_creator:
                    # Add parent context to specification
                    if parent_context:
                        action["specification"]["parent_output_structure"] = parent_context
                    
                    # Add user request for better context
                    if hasattr(tree, 'user_request'):
                        action["specification"]["user_request"] = tree.user_request
                    
                    block = self.function_creator.create_function_block(
                        action["specification"]
                    )
                    if block:
                        function_blocks.append(block)
            
            elif action["type"] == "use_existing":
                # Use function selector to find existing block
                if self.function_selector:
                    block = self.function_selector.select_function_block(
                        action["requirements"]
                    )
                    if block:
                        function_blocks.append(block)
        
        return function_blocks
    
    def _save_function_block_to_dir(self, block: Any, function_block_dir: Path):
        """Save function block code, config, and requirements to directory."""
        import json
        
        # Save code.py
        if hasattr(block, 'code'):
            code_file = function_block_dir / "code.py"
            with open(code_file, 'w') as f:
                f.write(block.code)
        
        # Save requirements.txt
        if hasattr(block, 'requirements'):
            req_file = function_block_dir / "requirements.txt"
            with open(req_file, 'w') as f:
                f.write(block.requirements)
        
        # Save config.json
        config_data = {
            "name": block.name,
            "type": block.type.value if hasattr(block.type, 'value') else str(block.type),
            "description": block.description,
            "parameters": block.parameters if hasattr(block, 'parameters') else {},
            "static_config": block.static_config.model_dump() if hasattr(block, 'static_config') and hasattr(block.static_config, 'model_dump') else {}
        }
        
        config_file = function_block_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def _log_function_creation_to_node(
        self,
        node: AnalysisNode,
        tree: AnalysisTree,
        output_dir: Path,
        block: Any,
        iteration: int,
        block_index: int
    ):
        """Log function creation details to the node's agent_tasks folder and save function_block."""
        try:
            # Create node directory structure
            tree_dir = output_dir / tree.id
            nodes_dir = tree_dir / "nodes"
            node_dir = nodes_dir / f"node_{node.id}"
            
            # Create all required subdirectories
            agent_tasks_dir = node_dir / "agent_tasks" / "function_creator"
            agent_tasks_dir.mkdir(parents=True, exist_ok=True)
            
            function_block_dir = node_dir / "function_block"
            function_block_dir.mkdir(parents=True, exist_ok=True)
            
            # Create other required directories
            jobs_dir = node_dir / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            
            outputs_dir = node_dir / "outputs"
            outputs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save function block code, config, and requirements
            self._save_function_block_to_dir(block, function_block_dir)
            
            # Create initial node_info.json
            from datetime import datetime
            node_info = {
                "id": node.id,
                "name": node.function_block.name,
                "type": node.function_block.type.value if hasattr(node.function_block.type, 'value') else str(node.function_block.type),
                "parent_id": node.parent_id,
                "children_ids": node.children if hasattr(node, 'children') else [],
                "state": node.state.value if hasattr(node.state, 'value') else str(node.state),
                "created_at": datetime.now().isoformat(),
                "level": node.level,
                "debug_attempts": node.debug_attempts
            }
            node_info_file = node_dir / "node_info.json"
            import json
            with open(node_info_file, 'w') as f:
                json.dump(node_info, f, indent=2)
            
            # Create log entry
            timestamp = datetime.now()
            log_data = {
                "task_id": f"create_{iteration:02d}_{block_index:02d}",
                "task_type": "function_creation",
                "agent": "function_creator",
                "timestamp": timestamp.isoformat(),
                "node_id": node.id,
                "iteration": iteration,
                "function_block": {
                    "name": block.name,
                    "type": block.type.value if hasattr(block.type, 'value') else str(block.type),
                    "description": block.description,
                    "parameters": block.parameters if hasattr(block, 'parameters') else {}
                },
                "metadata": {
                    "tree_id": tree.id,
                    "user_request": tree.user_request[:200] if hasattr(tree, 'user_request') else ""
                }
            }
            
            # Save log file
            log_file = agent_tasks_dir / f"creation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            # Save code version
            if hasattr(block, 'code'):
                code_dir = agent_tasks_dir / "code_versions"
                code_dir.mkdir(exist_ok=True)
                code_file = code_dir / f"v{iteration}_{block_index:02d}_{timestamp.strftime('%Y%m%d_%H%M%S')}.py"
                with open(code_file, 'w') as f:
                    f.write(block.code)
            
            logger.debug(f"Logged function creation to {log_file}")
            
        except Exception as e:
            logger.warning(f"Failed to log function creation: {e}")
    
    def _find_parent_for_block(
        self,
        tree: AnalysisTree,
        block: Union[NewFunctionBlock, ExistingFunctionBlock],
        recommendations: Dict[str, Any]
    ) -> Optional[str]:
        """Find appropriate parent node for a new block."""
        # Check if recommendations specify a parent
        parent_hint = recommendations.get("parent_node_id")
        if parent_hint and parent_hint in tree.nodes:
            return parent_hint
        
        # Otherwise, find a leaf node (node with no children)
        for node_id, node in tree.nodes.items():
            if len(node.children) == 0 and node.state == NodeState.COMPLETED:
                return node_id
        
        # If no completed leaf nodes, return any leaf node
        for node_id, node in tree.nodes.items():
            if len(node.children) == 0:
                return node_id
        
        return None
    
    def _execute_pending_nodes(
        self,
        tree: AnalysisTree,
        input_path: Path,
        output_dir: Path,
        verbose: bool
    ):
        """Execute nodes that are in pending state."""
        for node_id, node in tree.nodes.items():
            if node.state == NodeState.PENDING:
                # Determine input for this node
                if node.parent_id:
                    parent = tree.nodes.get(node.parent_id)
                    if parent and parent.output_data_id:
                        node_input = Path(parent.output_data_id)
                    else:
                        continue  # Parent not ready
                else:
                    node_input = input_path
                
                # Execute node
                if verbose:
                    logger.info(f"Executing node: {node.function_block.name}")
                
                state, output_path = self.node_executor.execute_node(
                    node=node,
                    tree=tree,
                    input_path=node_input,
                    output_base_dir=output_dir
                )
                
                # Update tree state
                self.tree_manager.update_node_state(node_id, state, output_data_id=output_path)
                
                # Handle failures with bug fixer
                if state == NodeState.FAILED and self.bug_fixer:
                    self._attempt_fix(node, tree, node_input, output_dir, verbose)
    
    def _attempt_fix(
        self,
        node: AnalysisNode,
        tree: AnalysisTree,
        input_path: Path,
        output_dir: Path,
        verbose: bool
    ):
        """Attempt to fix a failed node using bug fixer agent."""
        # Determine node directory for logging
        tree_dir = output_dir / tree.id
        nodes_dir = tree_dir / "nodes"
        node_dir = nodes_dir / f"node_{node.id}"
        
        for attempt in range(tree.max_debug_trials):
            if verbose:
                logger.info(f"Attempting fix for {node.function_block.name} (attempt {attempt + 1})")
            
            # Try to read actual error logs from the last job
            stdout_content = ""
            stderr_content = ""
            
            # Find the most recent job directory
            jobs_dir = node_dir / "jobs"
            if jobs_dir.exists():
                job_dirs = sorted([d for d in jobs_dir.iterdir() if d.is_dir()], 
                                key=lambda x: x.name, reverse=True)
                if job_dirs:
                    latest_job = job_dirs[0]
                    # Try to read stdout from past_jobs
                    past_jobs_dir = latest_job / "output" / "past_jobs"
                    if past_jobs_dir.exists():
                        failed_dirs = [d for d in past_jobs_dir.iterdir() if d.is_dir() and "failed" in d.name]
                        if failed_dirs:
                            latest_failed = sorted(failed_dirs, key=lambda x: x.name, reverse=True)[0]
                            stdout_file = latest_failed / "stdout.txt"
                            stderr_file = latest_failed / "stderr.txt"
                            
                            if stdout_file.exists():
                                with open(stdout_file, 'r') as f:
                                    stdout_content = f.read()
                            if stderr_file.exists():
                                with open(stderr_file, 'r') as f:
                                    stderr_content = f.read()
            
            # Get parent node's data structure if available
            parent_data_structure = None
            if node.parent_id:
                parent_node = tree.nodes.get(node.parent_id)
                if parent_node and parent_node.output_data_id:
                    # Try to read parent's data structure
                    parent_output_dir = Path(parent_node.output_data_id).parent
                    data_structure_file = parent_output_dir / "_data_structure.json"
                    if data_structure_file.exists():
                        try:
                            import json
                            with open(data_structure_file, 'r') as f:
                                parent_data_structure = json.load(f)
                        except Exception as e:
                            logger.debug(f"Could not read parent data structure: {e}")
            
            # Get error details with actual logs
            error_info = {
                "node_id": node.id,
                "error": node.error,
                "function_block": node.function_block,
                "node_dir": node_dir,  # Add node_dir for logging
                "error_message": str(node.error) if node.error else "Unknown error",
                "stdout": stdout_content,
                "stderr": stderr_content,
                "parent_data_structure": parent_data_structure  # Add parent context
            }
            
            # Use bug fixer to generate fix
            fix_result = self.bug_fixer.fix_error(error_info)
            
            if fix_result and fix_result.get('success') and fix_result.get('fixed_code'):
                # Apply fix to function block
                node.function_block.code = fix_result['fixed_code']
                if fix_result.get('fixed_requirements'):
                    node.function_block.requirements = fix_result['fixed_requirements']
                
                # Retry execution
                state, output_path = self.node_executor.execute_node(
                    node=node,
                    tree=tree,
                    input_path=input_path,
                    output_base_dir=output_dir
                )
                
                self.tree_manager.update_node_state(node.id, state, output_data_id=output_path)
                
                if state == NodeState.COMPLETED:
                    if verbose:
                        logger.info(f"Successfully fixed {node.function_block.name}")
                    break
    
    def _create_default_pipeline(self, tree: AnalysisTree):
        """Create a default pipeline for testing without LLM."""
        # Create a simple preprocessing block
        block = NewFunctionBlock(
            name="basic_preprocessing",
            type=FunctionBlockType.PYTHON,
            description="Basic data preprocessing",
            code="""
def run(path_dict, params):
    import scanpy as sc
    import os
    
    # Load data
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        import glob
        h5ad_files = glob.glob(os.path.join(path_dict["input_dir"], "*.h5ad"))
        if h5ad_files:
            input_path = h5ad_files[0]
    
    adata = sc.read_h5ad(input_path)
    
    # Basic processing
    min_genes = params.get('min_genes', 200)
    min_cells = params.get('min_cells', 3)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Save
    output_path = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    adata.write(output_path)
    return adata
""",
            static_config=StaticConfig(
                description="Basic preprocessing",
                tag="preprocessing",
                args=[]
            ),
            requirements="scanpy\nnumpy",
            parameters={}
        )
        
        self.tree_manager.add_root_node(block)
    
    def _execute_tree(
        self,
        tree: AnalysisTree,
        input_path: Path,
        output_dir: Path,
        verbose: bool
    ) -> Dict[str, Any]:
        """Execute all nodes in the analysis tree."""
        results = {}
        
        # Execute nodes level by level
        max_level = max((node.level for node in tree.nodes.values()), default=0)
        
        for level in range(max_level + 1):
            level_nodes = [
                (node_id, node) 
                for node_id, node in tree.nodes.items() 
                if node.level == level
            ]
            
            for node_id, node in level_nodes:
                if node.state != NodeState.PENDING:
                    continue
                
                # Determine input
                if node.parent_id:
                    parent = tree.nodes.get(node.parent_id)
                    if parent and parent.output_data_id:
                        node_input = Path(parent.output_data_id)
                    else:
                        continue
                else:
                    node_input = input_path
                
                # Execute
                if verbose:
                    logger.info(f"Executing: {node.function_block.name}")
                
                state, output_path = self.node_executor.execute_node(
                    node=node,
                    tree=tree,
                    input_path=node_input,
                    output_base_dir=output_dir
                )
                
                self.tree_manager.update_node_state(node_id, state, output_data_id=output_path)
                
                results[node_id] = {
                    "name": node.function_block.name,
                    "state": state.value,
                    "output": output_path,
                    "error": node.error if state == NodeState.FAILED else None
                }
                
                if state == NodeState.FAILED:
                    logger.error(f"  ✗ Failed: {node.function_block.name}")
                elif verbose:
                    logger.info(f"  ✓ Completed: {node.function_block.name}")
        
        return results