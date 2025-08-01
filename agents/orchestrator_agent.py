"""Orchestrator agent that coordinates the overall analysis workflow."""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_agent import BaseAgent
from .function_selector_agent import FunctionSelectorAgent
from .bug_fixer_agent import BugFixerAgent
from ..models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    NewFunctionBlock
)
from ..analysis_tree_management import AnalysisTreeManager
from ..utils.data_handler import DataHandler


class OrchestratorAgent(BaseAgent):
    """Agent that orchestrates the entire analysis workflow."""
    
    def __init__(
        self, 
        tree_manager: AnalysisTreeManager,
        function_selector: FunctionSelectorAgent,
        bug_fixer: BugFixerAgent
    ):
        super().__init__("orchestrator")
        self.tree_manager = tree_manager
        self.function_selector = function_selector
        self.bug_fixer = bug_fixer
        self.data_handler = DataHandler()
        
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the analysis workflow.
        
        Required context keys:
            - tree: AnalysisTree
            - user_request: str
            - max_iterations: int
            
        Returns:
            Dict with:
                - satisfied: bool
                - iterations: int
                - final_state: str
        """
        self.validate_context(context, ['tree', 'user_request', 'max_iterations'])
        
        tree = context['tree']
        user_request = context['user_request']
        max_iterations = context['max_iterations']
        
        satisfied = False
        iteration = 0
        
        while not satisfied and iteration < max_iterations:
            iteration += 1
            
            # Check if we need to generate new nodes
            if self._needs_expansion(tree):
                satisfied = self._expand_tree(tree, user_request, context)
            
            # Check if we have nodes to execute
            pending_nodes = self._get_pending_nodes(tree)
            if pending_nodes:
                self._mark_nodes_for_execution(pending_nodes)
            
            # Check if we're done
            if satisfied or not self.tree_manager.can_continue_expansion():
                break
                
        return {
            'satisfied': satisfied,
            'iterations': iteration,
            'final_state': self._get_tree_state(tree)
        }
    
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
        
        # Select root function block
        result = self.function_selector.process({
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
        
        # Select child function blocks
        result = self.function_selector.process({
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