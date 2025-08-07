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
from .function_creator_agent import FunctionCreatorAgent
from .bug_fixer_agent import BugFixerAgent

logger = setup_logger(__name__)


class MainAgent:
    """Main agent that coordinates the entire analysis workflow."""
    
    def __init__(self, openai_api_key: Optional[str] = None, llm_model: Optional[str] = None):
        """Initialize the main agent.
        
        Args:
            openai_api_key: OpenAI API key for LLM services
            llm_model: OpenAI model to use (e.g., 'gpt-4o', 'gpt-4o-mini')
        """
        # Initialize LLM service
        self.llm_service = OpenAIService(api_key=openai_api_key, model=llm_model) if openai_api_key else None
        
        # Initialize managers
        self.tree_manager = AnalysisTreeManager()
        self.executor_manager = ExecutorManager()
        self.node_executor = NodeExecutor(self.executor_manager)
        
        # Initialize specialized agents
        self.orchestrator = OrchestratorAgent(self.llm_service) if self.llm_service else None
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
        
        # Save agent info with hyperparameters
        agent_info = {
            "agent_type": "main",
            "tree_id": tree.id,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "created_at": datetime.now().isoformat(),
            "llm_model": llm_model,
            "generation_mode": generation_mode,
            "hyperparameters": {
                "max_nodes": max_nodes,
                "max_children": max_children,
                "max_debug_trials": max_debug_trials,
                "max_iterations": max_iterations
            }
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
            tree_md_path = output_dir / "directory_tree.md"
            self.tree_manager._create_directory_tree_md(output_dir, tree_md_path)
        
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
                logger.info(f"  Satisfied: {satisfied}, Total nodes: {tree.total_nodes}, Max nodes: {tree.max_nodes}")
            
            # First iteration: Create root node
            if tree.total_nodes == 0:
                # Get initial recommendation for root node using unified function creator
                creator_context = {
                    "user_request": tree.user_request,
                    "tree": tree,
                    "current_node": None,
                    "parent_chain": [],
                    "generation_mode": tree.generation_mode,
                    "max_children": 1,  # Root node should be single
                    "data_summary": {},  # Could load data summary here if needed
                    "is_root_node": True  # Explicit flag for root node
                }
                
                result = self.function_creator.process_selection_or_creation(creator_context)
                satisfied = result.get("satisfied", False)
                function_blocks = result.get("function_blocks", [])
                
                if verbose:
                    logger.info(f"  Function creator result: satisfied={satisfied}, blocks={len(function_blocks)}")
                
                # Create root node
                try:
                    
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
                    
                    # Check if user request is satisfied and get next function blocks
                    parent_chain = self.tree_manager.get_parent_chain(parent_id)
                    
                    creator_context = {
                        "user_request": tree.user_request,
                        "tree": tree,
                        "current_node": parent_node,
                        "parent_chain": parent_chain,
                        "generation_mode": tree.generation_mode,
                        "max_children": tree.max_children_per_node,
                        "data_summary": self._get_parent_data_summary(parent_node, output_dir),
                        "is_root_node": False  # Not a root node
                    }
                    
                    result = self.function_creator.process_selection_or_creation(creator_context)
                    satisfied = result.get("satisfied", False)
                    function_blocks = result.get("function_blocks", [])
                    
                    if verbose:
                        logger.info(f"  Creator result for {parent_node.function_block.name}: satisfied={satisfied}, blocks={len(function_blocks)}")
                    
                    if satisfied:
                        if verbose:
                            logger.info("User request satisfied")
                        break
                    
                    # Process function blocks for this parent
                    try:
                        
                        # Add children to this successful parent
                        for i, block in enumerate(function_blocks):
                            if tree.total_nodes >= tree.max_nodes:
                                break
                            
                            # Check if conversion is needed
                            conversion_block = self._check_conversion_needed(parent_node, block, output_dir)
                            
                            if conversion_block:
                                # First add conversion node
                                if verbose:
                                    logger.info(f"Adding conversion node: {conversion_block.name}")
                                
                                conv_nodes = self.tree_manager.add_child_nodes(parent_id, [conversion_block])
                                if conv_nodes:
                                    conv_node = conv_nodes[0]
                                    
                                    # Execute conversion node
                                    success = self._execute_single_node(conv_node, tree, input_path, output_dir, verbose)
                                    
                                    if success:
                                        # Now add the actual child to the conversion node
                                        parent_id = conv_node.id
                                        parent_node = conv_node
                                    else:
                                        logger.error(f"Conversion node {conversion_block.name} failed")
                                        continue
                            
                            # Add the actual child node
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
                # For now, we don't have existing blocks - log warning
                logger.warning(f"Existing block requested but not implemented: {action.get('name', 'unknown')}")
        
        return function_blocks
    
    def _get_parent_data_summary(self, parent_node: AnalysisNode, output_dir: Path) -> Dict[str, Any]:
        """Get data summary from parent node's output."""
        if not parent_node.output_data_id:
            return {}
        
        try:
            # Try to read data structure file
            parent_output_dir = Path(parent_node.output_data_id).parent
            data_structure_file = parent_output_dir / "_data_structure.json"
            
            if data_structure_file.exists():
                with open(data_structure_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Could not read parent data structure: {e}")
        
        return {}
    
    def _collect_bug_fix_history(self, node_dir: Path) -> List[Dict[str, Any]]:
        """Collect history of previous bug fix attempts for this node.
        
        Args:
            node_dir: Path to the node directory
            
        Returns:
            List of previous attempt information
        """
        history = []
        bug_fixes_dir = node_dir / "agent_tasks" / "bug_fixer" / "bug_fixes"
        
        if not bug_fixes_dir.exists():
            return history
            
        try:
            # Get all fix attempt files sorted by timestamp
            fix_files = sorted(bug_fixes_dir.glob("fix_attempt_*.json"))
            
            for fix_file in fix_files:
                with open(fix_file, 'r') as f:
                    attempt_data = json.load(f)
                    
                # Extract relevant information for history context
                history_entry = {
                    "attempt_number": attempt_data.get("attempt_number", 0),
                    "timestamp": attempt_data.get("timestamp", ""),
                    "error_info": attempt_data.get("error_info", {}),
                    "llm_analysis": attempt_data.get("llm_output", {}).get("analysis", {}),
                    "changes_made": attempt_data.get("llm_output", {}).get("changes_made", []),
                    "requirements_changes": attempt_data.get("llm_output", {}).get("requirements_changes", []),
                    "success": attempt_data.get("success", False),
                    "error": attempt_data.get("error"),
                    "fixed_code_snippet": self._extract_code_snippet(
                        attempt_data.get("fixed_code", ""),
                        attempt_data.get("llm_output", {}).get("analysis", {})
                    )
                }
                history.append(history_entry)
                
        except Exception as e:
            logger.warning(f"Error collecting bug fix history: {e}")
            
        return history
    
    def _extract_code_snippet(self, fixed_code: str, analysis: Dict[str, Any]) -> str:
        """Extract the most relevant part of the fix for context.
        
        Args:
            fixed_code: The full fixed code
            analysis: The LLM's analysis of the fix
            
        Returns:
            A snippet showing the key changes
        """
        # For now, just return the analysis summary
        # In future, could do more sophisticated extraction
        if analysis:
            return f"Root cause: {analysis.get('root_cause', 'Unknown')}, Strategy: {analysis.get('fix_strategy', 'Unknown')}"
        return "No analysis available"
    
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
                    # Try to read stdout/stderr from the job logs
                    logs_dir = latest_job / "logs"
                    if logs_dir.exists():
                        stdout_file = logs_dir / "stdout.txt"
                        stderr_file = logs_dir / "stderr.txt"
                        
                        if stdout_file.exists():
                            with open(stdout_file, 'r') as f:
                                stdout_content = f.read()
                                # If stdout is too long, get the tail with traceback
                                if len(stdout_content) > 10000:
                                    lines = stdout_content.split('\n')
                                    # Try to find where the error starts
                                    error_start = -1
                                    for i in range(len(lines) - 1, -1, -1):
                                        if 'Traceback (most recent call last):' in lines[i]:
                                            error_start = i
                                            break
                                    
                                    if error_start >= 0:
                                        # Include from traceback to end
                                        stdout_content = '\n'.join(lines[error_start:])
                                    else:
                                        # Just get last 200 lines if no traceback found
                                        stdout_content = '\n'.join(lines[-200:])
                                        stdout_content = f"[... stdout truncated, showing last 200 lines ...]\n{stdout_content}"
                                        
                        if stderr_file.exists():
                            with open(stderr_file, 'r') as f:
                                stderr_content = f.read()
                                # If stderr is too long, get the tail
                                if len(stderr_content) > 5000:
                                    lines = stderr_content.split('\n')
                                    stderr_content = '\n'.join(lines[-100:])
                                    stderr_content = f"[... stderr truncated, showing last 100 lines ...]\n{stderr_content}"
            
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
            
            # Collect previous bug fix attempts history
            previous_attempts = self._collect_bug_fix_history(node_dir)
            
            # Get error details with actual logs
            error_info = {
                "node_id": node.id,
                "error": node.error,
                "function_block": node.function_block,
                "node_dir": node_dir,  # Add node_dir for logging
                "error_message": str(node.error) if node.error else "Unknown error",
                "stdout": stdout_content,
                "stderr": stderr_content,
                "parent_data_structure": parent_data_structure,  # Add parent context
                "previous_attempts": previous_attempts  # Add history
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
    
    def _check_conversion_needed(
        self,
        parent_node: AnalysisNode,
        child_block: Union[NewFunctionBlock, ExistingFunctionBlock],
        output_dir: Path
    ) -> Optional[Union[NewFunctionBlock, ExistingFunctionBlock]]:
        """Check if conversion is needed between parent and child nodes.
        
        Returns conversion function block if needed, None otherwise.
        """
        if not parent_node or not parent_node.output_data_id:
            return None
        
        # The output_data_id points to the outputs directory
        parent_output_dir = Path(parent_node.output_data_id)
        
        # Check parent output files
        has_anndata = (parent_output_dir / "_node_anndata.h5ad").exists()
        has_seurat = (parent_output_dir / "_node_seuratObject.rds").exists()
        has_sc_matrix = (parent_output_dir / "_node_sc_matrix").exists()
        
        # If sc_matrix exists, no conversion needed
        if has_sc_matrix:
            return None
        
        # Determine parent output type
        parent_type = None
        if has_anndata:
            parent_type = FunctionBlockType.PYTHON
        elif has_seurat:
            parent_type = FunctionBlockType.R
        else:
            return None  # No recognized output
        
        # If types match, no conversion needed
        if parent_type == child_block.type:
            return None
        
        # Create appropriate conversion block
        if parent_type == FunctionBlockType.PYTHON and child_block.type == FunctionBlockType.R:
            # Python to R: use convert_anndata_to_sc_matrix
            # Read the conversion code from the builtin file
            conversion_code_path = Path(__file__).parent.parent / "src/ragomics_agent_local/function_blocks/builtin/convert_anndata_to_sc_matrix.py"
            if conversion_code_path.exists():
                with open(conversion_code_path, 'r') as f:
                    code = f.read()
            else:
                # Inline the conversion code
                code = '''
def run(path_dict, params):
    """Convert AnnData to shared single-cell matrix format."""
    import os
    import json
    import anndata
    import pandas as pd
    import numpy as np
    from scipy.io import mmwrite
    from scipy.sparse import issparse
    from pathlib import Path
    
    input_dir = Path(path_dict['input_dir'])
    output_dir = Path(path_dict['output_dir'])
    
    # Find AnnData file
    h5ad_files = list(input_dir.glob('_node_anndata.h5ad'))
    if not h5ad_files:
        raise FileNotFoundError("No _node_anndata.h5ad file found in input directory")
    
    adata_path = h5ad_files[0]
    print(f"Loading AnnData from: {adata_path}")
    
    # Load AnnData
    adata = anndata.read_h5ad(adata_path)
    print(f"Loaded AnnData with shape: {adata.shape}")
    
    # Create output structure
    sc_matrix_dir = output_dir / '_node_sc_matrix'
    sc_matrix_dir.mkdir(exist_ok=True)
    
    # Helper function to write matrix
    def write_matrix(matrix, path, name):
        """Write matrix in appropriate format."""
        if issparse(matrix):
            # Write as MTX format for sparse matrices
            mmwrite(str(path / f"{name}.mtx"), matrix)
            return {"type": "sparse", "format": "mtx", "shape": list(matrix.shape)}
        else:
            # Write as CSV for dense matrices
            if isinstance(matrix, pd.DataFrame):
                matrix.to_csv(path / f"{name}.csv", index=False)
            else:
                np.savetxt(path / f"{name}.csv", matrix, delimiter=',')
            return {"type": "dense", "format": "csv", "shape": list(matrix.shape)}
    
    # Create metadata dictionary
    metadata = {
        "source": "anndata",
        "shape": list(adata.shape),
        "components": {}
    }
    
    # 1. Write cell and gene names
    with open(sc_matrix_dir / 'obs_names.txt', 'w') as f:
        for name in adata.obs_names:
            f.write(f"{name}\\n")
    
    with open(sc_matrix_dir / 'var_names.txt', 'w') as f:
        for name in adata.var_names:
            f.write(f"{name}\\n")
    
    # 2. Write main expression matrix (X)
    x_info = write_matrix(adata.X, sc_matrix_dir, 'X')
    metadata['components']['X'] = x_info
    
    # 3. Write obs (cell metadata) if present
    if len(adata.obs.columns) > 0:
        obs_dir = sc_matrix_dir / 'obs'
        obs_dir.mkdir(exist_ok=True)
        
        obs_info = {}
        for col in adata.obs.columns:
            series = adata.obs[col]
            series.to_csv(obs_dir / f"{col}.csv", index=False, header=[col])
            obs_info[col] = {
                "dtype": str(series.dtype),
                "shape": len(series)
            }
        metadata['components']['obs'] = obs_info
    
    # 4. Write metadata file
    with open(sc_matrix_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy original files to output
    import shutil
    shutil.copy2(adata_path, output_dir / '_node_anndata.h5ad')
    
    # Check if Seurat object exists and copy it
    rds_files = list(input_dir.glob('_node_seuratObject.rds'))
    if rds_files:
        shutil.copy2(rds_files[0], output_dir / '_node_seuratObject.rds')
    
    print(f"Successfully converted AnnData to _node_sc_matrix format")
    print(f"Output directory: {sc_matrix_dir}")
    
    return adata
'''
            
            return NewFunctionBlock(
                name="convert_anndata_to_sc_matrix",
                type=FunctionBlockType.PYTHON,
                description="Convert AnnData to shared SC matrix format for R processing",
                code=code,
                requirements="anndata\nscanpy\nscipy\npandas\nnumpy",
                parameters={},
                static_config=StaticConfig(
                    args=[],
                    description="Convert AnnData to SC matrix",
                    tag="conversion"
                )
            )
            
        elif parent_type == FunctionBlockType.R and child_block.type == FunctionBlockType.PYTHON:
            # R to Python: use convert_seurat_to_sc_matrix
            # Read the conversion code from the builtin file
            conversion_code_path = Path(__file__).parent.parent / "src/ragomics_agent_local/function_blocks/builtin/convert_seurat_to_sc_matrix.r"
            if conversion_code_path.exists():
                with open(conversion_code_path, 'r') as f:
                    code = f.read()
            else:
                # Inline the conversion code
                code = '''
run <- function(path_dict, params) {
    # Load required libraries
    library(Seurat)
    library(Matrix)
    library(jsonlite)
    
    input_dir <- path_dict$input_dir
    output_dir <- path_dict$output_dir
    
    # Find Seurat object file
    rds_files <- list.files(input_dir, pattern = "_node_seuratObject\\\\.rds$", full.names = TRUE)
    if (length(rds_files) == 0) {
        stop("No _node_seuratObject.rds file found in input directory")
    }
    
    seurat_path <- rds_files[1]
    cat("Loading Seurat object from:", seurat_path, "\\n")
    
    # Load Seurat object
    srt <- readRDS(seurat_path)
    cat("Loaded Seurat object with", ncol(srt), "cells and", nrow(srt), "features\\n")
    
    # Create output structure
    sc_matrix_dir <- file.path(output_dir, "_node_sc_matrix")
    dir.create(sc_matrix_dir, showWarnings = FALSE)
    
    # Helper function to write matrix
    write_matrix <- function(mat, path, name) {
        if (inherits(mat, "sparseMatrix")) {
            # Write as MTX format for sparse matrices
            writeMM(mat, file.path(path, paste0(name, ".mtx")))
            return(list(type = "sparse", format = "mtx", shape = dim(mat)))
        } else {
            # Write as CSV for dense matrices
            write.csv(mat, file.path(path, paste0(name, ".csv")), row.names = FALSE)
            return(list(type = "dense", format = "csv", shape = dim(mat)))
        }
    }
    
    # Create metadata list
    metadata <- list(
        source = "seurat",
        shape = c(ncol(srt), nrow(srt)),
        components = list()
    )
    
    # 1. Write cell and gene names
    writeLines(colnames(srt), file.path(sc_matrix_dir, "obs_names.txt"))
    writeLines(rownames(srt), file.path(sc_matrix_dir, "var_names.txt"))
    
    # 2. Write main expression matrix (use counts or data)
    # Get the default assay
    default_assay <- DefaultAssay(srt)
    assay_obj <- srt[[default_assay]]
    
    # Try to get counts first, then data
    if (length(GetAssayData(assay_obj, slot = "counts")) > 0) {
        X <- GetAssayData(assay_obj, slot = "counts")
    } else {
        X <- GetAssayData(assay_obj, slot = "data")
    }
    
    # Transpose to match AnnData format (cells x genes)
    X <- t(X)
    x_info <- write_matrix(X, sc_matrix_dir, "X")
    metadata$components$X <- x_info
    
    # 3. Write metadata file
    write(toJSON(metadata, pretty = TRUE, auto_unbox = TRUE), 
          file.path(sc_matrix_dir, "metadata.json"))
    
    # Copy original files to output
    file.copy(seurat_path, file.path(output_dir, "_node_seuratObject.rds"))
    
    # Check if AnnData exists and copy it
    h5ad_files <- list.files(input_dir, pattern = "_node_anndata\\\\.h5ad$", full.names = TRUE)
    if (length(h5ad_files) > 0) {
        file.copy(h5ad_files[1], file.path(output_dir, "_node_anndata.h5ad"))
    }
    
    cat("Successfully converted Seurat object to _node_sc_matrix format\\n")
    cat("Output directory:", sc_matrix_dir, "\\n")
    
    return(srt)
}
'''
            
            return NewFunctionBlock(
                name="convert_seurat_to_sc_matrix",
                type=FunctionBlockType.R,
                description="Convert Seurat object to shared SC matrix format for Python processing",
                code=code,
                requirements="Seurat\nMatrix\njsonlite",
                parameters={},
                static_config=StaticConfig(
                    args=[],
                    description="Convert Seurat to SC matrix",
                    tag="conversion"
                )
            )
        
        return None