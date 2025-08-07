"""Orchestrator agent that coordinates the overall analysis workflow with parallel execution."""

from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime
import uuid
import json

from .base_agent import BaseAgent
from .function_creator_agent import FunctionCreatorAgent
from .bug_fixer_agent import BugFixerAgent
from ..models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    NewFunctionBlock
)
from ..analysis_tree_management import AnalysisTreeManager
from ..utils.data_handler import DataHandler
from ..job_executors.job_pool import JobPool, JobResult, JobStatus, JobPriority


class OrchestratorAgent(BaseAgent):
    """Agent that orchestrates the entire analysis workflow with parallel execution."""
    
    def __init__(
        self, 
        llm_service=None,
        tree_manager: Optional[AnalysisTreeManager] = None,
        function_creator: Optional[FunctionCreatorAgent] = None,
        bug_fixer: Optional[BugFixerAgent] = None,
        max_parallel_jobs: int = 3
    ):
        super().__init__("orchestrator")
        self.tree_manager = tree_manager
        self.function_creator = function_creator
        self.bug_fixer = bug_fixer
        self.data_handler = DataHandler()
        self.max_parallel_jobs = max_parallel_jobs
        
        # Job pool for parallel execution
        self.job_pool: Optional[JobPool] = None
        
        # State tracking
        self.completed_nodes: Set[str] = set()
        self.failed_nodes: Set[str] = set()
        self.node_results: Dict[str, Dict[str, Any]] = {}
        self.expansion_decisions: Dict[str, Dict[str, Any]] = {}
        
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the analysis workflow with parallel execution.
        
        Required context keys:
            - tree: AnalysisTree
            - user_request: str
            - max_iterations: int
            - node_executor: NodeExecutor (optional)
            - output_dir: Path (optional)
            
        Returns:
            Dict with:
                - satisfied: bool
                - iterations: int
                - final_state: str
                - completed_nodes: List[str]
                - failed_nodes: List[str]
        """
        self.validate_context(context, ['tree', 'user_request', 'max_iterations'])
        
        tree = context['tree']
        user_request = context['user_request']
        max_iterations = context['max_iterations']
        node_executor = context.get('node_executor')
        output_dir = context.get('output_dir', Path('output'))
        
        # Initialize job pool
        self.job_pool = JobPool(
            max_parallel_jobs=self.max_parallel_jobs,
            executor_type="thread",
            enable_callbacks=True
        )
        
        # Reset state
        self.completed_nodes.clear()
        self.failed_nodes.clear()
        self.node_results.clear()
        self.expansion_decisions.clear()
        
        satisfied = False
        iteration = 0
        
        try:
            while not satisfied and iteration < max_iterations:
                iteration += 1
                
                # Check if we need to generate new nodes
                if self._needs_expansion(tree):
                    satisfied = self._expand_tree(tree, user_request, context)
                
                # Submit pending nodes for parallel execution
                pending_nodes = self._get_pending_nodes(tree)
                if pending_nodes and node_executor:
                    self._submit_nodes_for_execution(
                        pending_nodes, tree, node_executor, output_dir
                    )
                
                # Wait for some nodes to complete before continuing
                if self.job_pool.job_queue.qsize() > 0 or len(self.job_pool.running_jobs) > 0:
                    self._wait_for_node_completion(timeout=5.0)
                
                # Check if we're done
                if satisfied or not self.tree_manager.can_continue_expansion():
                    break
            
            # Wait for all remaining jobs to complete
            if node_executor:
                self.logger.info("Waiting for all jobs to complete...")
                all_results = self.job_pool.wait_for_all(timeout=300)
                self.logger.info(f"All jobs completed: {len(all_results)} total")
                
        finally:
            # Cleanup
            if self.job_pool:
                self.job_pool.shutdown(wait=True)
                
        return {
            'satisfied': satisfied,
            'iterations': iteration,
            'final_state': self._get_tree_state(tree),
            'completed_nodes': list(self.completed_nodes),
            'failed_nodes': list(self.failed_nodes)
        }
    
    def _submit_nodes_for_execution(
        self,
        nodes: List[AnalysisNode],
        tree: AnalysisTree,
        node_executor,
        output_dir: Path
    ):
        """Submit nodes for parallel execution."""
        for node in nodes:
            if node.id in self.node_results:
                continue  # Already submitted
            
            # Determine dependencies
            dependencies = set()
            if node.parent_id and node.parent_id not in self.completed_nodes:
                dependencies.add(f"node_{node.parent_id}")  # Job ID format
            
            # Submit job
            job_id = self.job_pool.submit_job(
                node_id=node.id,
                tree_id=tree.id,
                execute_fn=self._execute_node_wrapper,
                args=(node, tree, node_executor, output_dir),
                priority=JobPriority.NORMAL,
                callback=lambda result, n=node: self._handle_node_completion(n, result),
                dependencies=dependencies,
                metadata={
                    "node_name": node.function_block.name,
                    "node_level": node.level
                }
            )
            
            # Track submission
            self.node_results[node.id] = {
                "job_id": job_id,
                "status": NodeState.PENDING
            }
            
            self.logger.info(f"Submitted node {node.id} for execution (job_id={job_id})")
    
    def _execute_node_wrapper(
        self,
        node: AnalysisNode,
        tree: AnalysisTree,
        node_executor,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Wrapper for node execution to handle errors."""
        try:
            # Get input path from parent if available
            input_path = tree.input_data_path
            if node.parent_id and node.parent_id in self.node_results:
                parent_result = self.node_results[node.parent_id]
                if parent_result.get('output_path'):
                    input_path = parent_result['output_path']
            
            # Execute node
            state, output_path = node_executor.execute_node(
                node=node,
                tree=tree,
                input_path=input_path,
                output_base_dir=output_dir
            )
            
            return {
                "node_id": node.id,
                "state": state,
                "output_path": str(output_path) if output_path else None
            }
            
        except Exception as e:
            self.logger.error(f"Error executing node {node.id}: {e}")
            return {
                "node_id": node.id,
                "state": NodeState.FAILED,
                "error": str(e)
            }
    
    def _handle_node_completion(self, node: AnalysisNode, job_result: JobResult):
        """Handle node completion with reactive expansion."""
        self.logger.info(f"Node {node.id} completed with status {job_result.status}")
        
        if job_result.status == JobStatus.COMPLETED:
            result = job_result.result
            self.node_results[node.id].update(result)
            
            if result['state'] == NodeState.COMPLETED:
                self.completed_nodes.add(node.id)
                
                # Make reactive expansion decision
                self._make_expansion_decision(node, result)
                
            elif result['state'] == NodeState.FAILED:
                self.failed_nodes.add(node.id)
                # Try to fix if possible
                if hasattr(self, 'handle_failed_node'):
                    self.handle_failed_node(
                        node,
                        result.get('error', 'Unknown error'),
                        '', ''
                    )
        else:
            self.failed_nodes.add(node.id)
            self.node_results[node.id]['state'] = NodeState.FAILED
    
    def _make_expansion_decision(self, node: AnalysisNode, result: Dict[str, Any]):
        """Make decision about expanding from completed node."""
        # Check if node should be expanded
        if node.level >= 3:  # Max depth
            self.logger.info(f"Node {node.id} at max depth, not expanding")
            return
        
        # Simple heuristic: expand if node completed successfully
        # In future, use LLM to make intelligent decisions
        decision = {
            "should_expand": True,
            "reason": "Node completed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
        self.expansion_decisions[node.id] = decision
    
    def plan_next_steps(self, task: dict) -> dict:
        """Plan next steps for analysis workflow.
        
        Args:
            task: Dictionary with user_request, tree_state, iteration, etc.
            
        Returns:
            Dictionary with satisfied flag and next_actions list
        """
        user_request = task.get("user_request", "")
        tree_state = task.get("tree_state", {})
        iteration = task.get("iteration", 1)
        
        # Simple logic for testing - in real implementation, use LLM
        total_nodes = tree_state.get("total_nodes", 0)
        completed_nodes = tree_state.get("completed_nodes", 0)
        
        # Define simple workflow steps
        workflow_steps = [
            {
                "type": "create_new",
                "name": "quality_control",
                "specification": {
                    "name": "quality_control",
                    "description": "Filter cells and genes based on QC metrics",
                    "task": "Apply quality control filters to remove low-quality cells and genes"
                }
            },
            {
                "type": "create_new",
                "name": "normalization", 
                "specification": {
                    "name": "normalization",
                    "description": "Normalize and log-transform data",
                    "task": "Normalize expression data and apply log transformation"
                }
            },
            {
                "type": "create_new",
                "name": "dimensionality_reduction",
                "specification": {
                    "name": "dimensionality_reduction", 
                    "description": "Perform PCA and UMAP",
                    "task": "Apply PCA and UMAP for dimensionality reduction"
                }
            },
            {
                "type": "create_new",
                "name": "clustering_analysis",
                "specification": {
                    "name": "clustering_analysis",
                    "description": "Run multiple clustering methods", 
                    "task": "Apply various clustering algorithms and compare results"
                }
            },
            {
                "type": "create_new",
                "name": "metrics_evaluation",
                "specification": {
                    "name": "metrics_evaluation",
                    "description": "Calculate clustering metrics",
                    "task": "Compute clustering quality metrics and generate report"
                }
            }
        ]
        
        # Simple logic for testing - should be replaced with proper LLM-based planning
        # Do NOT hardcode satisfaction logic based on specific analysis types
        
        # NEVER hardcode a maximum number of nodes for satisfaction
        # The tree should expand based on user parameters (max_nodes, max_children, etc.)
        
        # For now, continue expanding until we hit the workflow steps limit
        # This is temporary logic - real implementation should use LLM
        if total_nodes >= len(workflow_steps):
            return {"satisfied": True, "next_actions": []}
        
        # Return next step
        if total_nodes < len(workflow_steps):
            next_step = workflow_steps[total_nodes]
            return {
                "satisfied": False,
                "next_actions": [next_step],
                "reasoning": f"Step {total_nodes + 1}: {next_step['name']}"
            }
        
        return {"satisfied": True, "next_actions": []}
    
    def _wait_for_node_completion(self, timeout: float = 5.0):
        """Wait for at least one node to complete."""
        import time
        start_time = time.time()
        initial_completed = len(self.completed_nodes)
        
        while time.time() - start_time < timeout:
            if len(self.completed_nodes) > initial_completed:
                break
            time.sleep(0.1)
    
    def _needs_expansion(self, tree: AnalysisTree) -> bool:
        """Check if the tree needs new nodes."""
        # Tree needs expansion if:
        # 1. No root node yet
        # 2. All leaf nodes are completed and have no children
        
        if not tree.root_node_id:
            return True
            
        leaf_nodes = self._get_leaf_nodes(tree)
        return all(
            node.state == NodeState.COMPLETED and not node.children
            for node in leaf_nodes
        )
    
    def _expand_tree(
        self, 
        tree: AnalysisTree, 
        user_request: str,
        context: Dict[str, Any]
    ) -> bool:
        """Expand the tree with new nodes.
        
        Returns:
            bool: Whether the user request is satisfied
        """
        satisfied = False
        
        if not tree.root_node_id:
            # Create root node
            satisfied = self._create_root_node(tree, user_request, context)
        else:
            # Expand leaf nodes
            leaf_nodes = self._get_completed_leaf_nodes(tree)
            max_children = context.get('max_children', 3)
            
            # Expand each completed leaf node
            for leaf_node in leaf_nodes[:max_children]:
                node_satisfied = self._expand_node(
                    tree, leaf_node, user_request, context
                )
                if node_satisfied:
                    satisfied = True
                    break
                    
        return satisfied
    
    def _create_root_node(
        self, 
        tree: AnalysisTree,
        user_request: str,
        context: Dict[str, Any]
    ) -> bool:
        """Create the root node of the tree."""
        # Get data summary
        data_path = context.get('input_data_path')
        data_summary = {}
        if data_path:
            try:
                adata = self.data_handler.load_data(Path(data_path))
                data_summary = self.data_handler.get_data_summary(adata)
            except Exception as e:
                self.logger.warning(f"Failed to load data: {e}")
        
        # Select or create root function block
        result = self.function_creator.process_selection_or_creation({
            'user_request': user_request,
            'tree': tree,
            'current_node': None,
            'parent_chain': [],
            'generation_mode': tree.generation_mode,
            'max_children': 1,  # Root should be single
            'data_path': data_path,
            'data_summary': data_summary
        })
        
        if result['function_blocks']:
            self.tree_manager.add_root_node(result['function_blocks'][0])
            
        return result['satisfied']
    
    def _expand_node(
        self,
        tree: AnalysisTree,
        node: AnalysisNode,
        user_request: str,
        context: Dict[str, Any]
    ) -> bool:
        """Expand a single node with children."""
        # Get parent chain
        parent_chain = self.tree_manager.get_parent_chain(node.id)
        
        # Get latest data path
        data_path = self.tree_manager.get_latest_data_path(node.id)
        
        # Select or create child function blocks
        result = self.function_creator.process_selection_or_creation({
            'user_request': user_request,
            'tree': tree,
            'current_node': node,
            'parent_chain': parent_chain,
            'generation_mode': tree.generation_mode,
            'max_children': context.get('max_children', 3),
            'data_path': data_path
        })
        
        if result['function_blocks']:
            self.tree_manager.add_child_nodes(node.id, result['function_blocks'])
            
        return result['satisfied']
    
    def handle_failed_node(
        self,
        node: AnalysisNode,
        error_message: str,
        stdout: str,
        stderr: str,
        max_debug_attempts: int = 3
    ) -> bool:
        """Handle a failed node by attempting to fix it.
        
        Returns:
            bool: Whether the fix was successful
        """
        if node.debug_attempts >= max_debug_attempts:
            return False
            
        if not isinstance(node.function_block, NewFunctionBlock):
            return False
            
        # Try to fix the code
        result = self.bug_fixer.process({
            'function_block': node.function_block,
            'error_message': error_message,
            'stdout': stdout,
            'stderr': stderr,
            'previous_attempts': []  # TODO: Track previous attempts
        })
        
        if result['success'] and result['fixed_code']:
            # Update the function block
            node.function_block.code = result['fixed_code']
            if result['fixed_requirements']:
                node.function_block.requirements = result['fixed_requirements']
            
            # Increment debug attempts
            self.tree_manager.increment_debug_attempts(node.id)
            
            # Mark as pending to retry
            self.tree_manager.update_node_execution(node.id, NodeState.PENDING)
            
            return True
            
        return False
    
    def _get_leaf_nodes(self, tree: AnalysisTree) -> List[AnalysisNode]:
        """Get all leaf nodes in the tree."""
        return [
            node for node in tree.nodes.values()
            if not node.children
        ]
    
    def _get_completed_leaf_nodes(self, tree: AnalysisTree) -> List[AnalysisNode]:
        """Get completed leaf nodes that can be expanded."""
        return [
            node for node in tree.nodes.values()
            if node.state == NodeState.COMPLETED and not node.children
        ]
    
    def _get_pending_nodes(self, tree: AnalysisTree) -> List[AnalysisNode]:
        """Get nodes that are pending execution."""
        return [
            node for node in tree.nodes.values()
            if node.state == NodeState.PENDING
        ]
    
    def _mark_nodes_for_execution(self, nodes: List[AnalysisNode]) -> None:
        """Mark nodes as ready for execution."""
        # In the current implementation, pending nodes are already
        # ready for execution. This method is here for future
        # enhancements where we might need additional logic.
        pass
    
    def _get_tree_state(self, tree: AnalysisTree) -> str:
        """Get a summary of the tree state."""
        total = len(tree.nodes)
        completed = sum(1 for n in tree.nodes.values() if n.state == NodeState.COMPLETED)
        failed = sum(1 for n in tree.nodes.values() if n.state == NodeState.FAILED)
        pending = sum(1 for n in tree.nodes.values() if n.state == NodeState.PENDING)
        
        return f"Total: {total}, Completed: {completed}, Failed: {failed}, Pending: {pending}"