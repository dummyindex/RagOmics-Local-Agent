"""Analysis tree manager for orchestrating hierarchical analysis."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    NewFunctionBlock, ExistingFunctionBlock
)
from ..utils.logger import get_logger
from ..utils.data_handler import DataHandler
from ..config import config

logger = get_logger(__name__)


class AnalysisTreeManager:
    """Manages the hierarchical analysis tree structure."""
    
    def __init__(self, tree: Optional[AnalysisTree] = None):
        self.tree = tree
        self.data_handler = DataHandler()
        
    def create_tree(
        self,
        user_request: str,
        input_data_path: str,
        max_nodes: int = 20,
        max_children_per_node: int = 3,
        max_debug_trials: int = 3,
        generation_mode: GenerationMode = GenerationMode.MIXED,
        llm_model: str = "gpt-4o-2024-08-06"
    ) -> AnalysisTree:
        """Create a new analysis tree."""
        
        self.tree = AnalysisTree(
            user_request=user_request,
            input_data_path=input_data_path,
            max_nodes=max_nodes,
            max_children_per_node=max_children_per_node,
            max_debug_trials=max_debug_trials,
            generation_mode=generation_mode,
            llm_model=llm_model
        )
        
        logger.info(f"Created new analysis tree: {self.tree.id}")
        return self.tree
    
    def add_root_node(self, function_block: Any) -> AnalysisNode:
        """Add root node to the tree."""
        
        if not self.tree:
            raise ValueError("No tree initialized")
            
        root_node = AnalysisNode(
            analysis_id=self.tree.id,
            function_block=function_block,
            level=0
        )
        
        self.tree.add_node(root_node)
        self.tree.root_node_id = root_node.id
        
        logger.info(f"Added root node: {root_node.id}")
        return root_node
    
    def add_child_nodes(
        self, 
        parent_node_id: str, 
        function_blocks: List[Any]
    ) -> List[AnalysisNode]:
        """Add child nodes to a parent node."""
        
        if not self.tree:
            raise ValueError("No tree initialized")
            
        parent_node = self.tree.get_node(parent_node_id)
        if not parent_node:
            raise ValueError(f"Parent node not found: {parent_node_id}")
        
        # Check constraints
        if len(parent_node.children) + len(function_blocks) > self.tree.max_children_per_node:
            logger.warning(
                f"Cannot add {len(function_blocks)} children to node {parent_node_id}. "
                f"Would exceed max children limit of {self.tree.max_children_per_node}"
            )
            function_blocks = function_blocks[:self.tree.max_children_per_node - len(parent_node.children)]
        
        if self.tree.total_nodes + len(function_blocks) > self.tree.max_nodes:
            logger.warning(
                f"Cannot add {len(function_blocks)} nodes. "
                f"Would exceed max nodes limit of {self.tree.max_nodes}"
            )
            function_blocks = function_blocks[:self.tree.max_nodes - self.tree.total_nodes]
        
        child_nodes = []
        for fb in function_blocks:
            child_node = AnalysisNode(
                parent_id=parent_node_id,
                analysis_id=self.tree.id,
                function_block=fb,
                level=parent_node.level + 1
            )
            
            self.tree.add_node(child_node)
            child_nodes.append(child_node)
            
        logger.info(f"Added {len(child_nodes)} child nodes to {parent_node_id}")
        return child_nodes
    
    def get_execution_order(self) -> List[AnalysisNode]:
        """Get nodes in execution order (breadth-first)."""
        
        if not self.tree or not self.tree.root_node_id:
            return []
        
        execution_order = []
        queue = [self.tree.root_node_id]
        
        while queue:
            node_id = queue.pop(0)
            node = self.tree.get_node(node_id)
            
            if node and node.state == NodeState.PENDING:
                execution_order.append(node)
                
            if node:
                queue.extend(node.children)
        
        return execution_order
    
    def get_parent_chain(self, node_id: str) -> List[AnalysisNode]:
        """Get all parent nodes up to root."""
        
        if not self.tree:
            return []
            
        chain = []
        current_id = node_id
        
        while current_id:
            node = self.tree.get_node(current_id)
            if not node:
                break
                
            if node.parent_id:  # Don't include the node itself
                parent = self.tree.get_node(node.parent_id)
                if parent:
                    chain.insert(0, parent)
                    current_id = node.parent_id
                else:
                    break
            else:
                break
        
        return chain
    
    def update_node_execution(
        self,
        node_id: str,
        state: NodeState,
        output_data_id: Optional[str] = None,
        figures: Optional[List[str]] = None,
        logs: Optional[List[str]] = None,
        error: Optional[str] = None,
        duration: Optional[float] = None
    ) -> None:
        """Update node after execution."""
        
        if not self.tree:
            return
            
        node = self.tree.get_node(node_id)
        if not node:
            return
        
        # Update state
        self.tree.update_node_state(node_id, state)
        
        # Update execution info
        if state == NodeState.RUNNING:
            node.start_time = datetime.now()
        elif state in [NodeState.COMPLETED, NodeState.FAILED]:
            node.end_time = datetime.now()
            if duration:
                node.duration = duration
            elif node.start_time:
                node.duration = (node.end_time - node.start_time).total_seconds()
        
        # Update results
        if output_data_id:
            node.output_data_id = output_data_id
        if figures:
            node.figures.extend(figures)
        if logs:
            node.logs.extend(logs)
        if error:
            node.error = error
        
        node.updated_at = datetime.now()
        
        logger.info(f"Updated node {node_id} to state {state}")
    
    def increment_debug_attempts(self, node_id: str) -> int:
        """Increment debug attempts for a node."""
        
        if not self.tree:
            return 0
            
        node = self.tree.get_node(node_id)
        if node:
            node.debug_attempts += 1
            return node.debug_attempts
        
        return 0
    
    def can_continue_expansion(self) -> bool:
        """Check if tree can continue expanding."""
        
        if not self.tree:
            return False
            
        # Check node limit
        if self.tree.total_nodes >= self.tree.max_nodes:
            logger.info("Reached maximum node limit")
            return False
        
        # Check if there are pending nodes
        has_pending = any(
            node.state == NodeState.PENDING 
            for node in self.tree.nodes.values()
        )
        
        return has_pending
    
    def get_latest_data_path(self, node_id: str) -> Optional[Path]:
        """Get the latest data path from node or its parents."""
        
        if not self.tree:
            return None
            
        # Check current node
        node = self.tree.get_node(node_id)
        if node and node.output_data_id:
            return Path(node.output_data_id)
        
        # Check parent chain
        parents = self.get_parent_chain(node_id)
        for parent in reversed(parents):
            if parent.output_data_id:
                return Path(parent.output_data_id)
        
        # Fall back to input data
        return Path(self.tree.input_data_path)
    
    def save_tree(self, path: Path) -> None:
        """Save tree to JSON file."""
        
        if not self.tree:
            return
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.tree.model_dump(), f, indent=2, default=str)
        
        logger.info(f"Saved analysis tree to {path}")
    
    def load_tree(self, path: Path) -> AnalysisTree:
        """Load tree from JSON file."""
        
        with open(path) as f:
            data = json.load(f)
        
        self.tree = AnalysisTree(**data)
        logger.info(f"Loaded analysis tree from {path}")
        
        return self.tree
    
    def get_summary(self) -> Dict[str, Any]:
        """Get tree execution summary."""
        
        if not self.tree:
            return {}
            
        # Calculate statistics
        completed_nodes = [n for n in self.tree.nodes.values() if n.state == NodeState.COMPLETED]
        failed_nodes = [n for n in self.tree.nodes.values() if n.state == NodeState.FAILED]
        total_duration = sum(n.duration or 0 for n in completed_nodes)
        
        return {
            "tree_id": self.tree.id,
            "user_request": self.tree.user_request,
            "total_nodes": self.tree.total_nodes,
            "completed_nodes": len(completed_nodes),
            "failed_nodes": len(failed_nodes),
            "pending_nodes": self.tree.total_nodes - len(completed_nodes) - len(failed_nodes),
            "total_duration_seconds": total_duration,
            "created_at": self.tree.created_at,
            "updated_at": self.tree.updated_at,
            "max_depth": max((n.level for n in self.tree.nodes.values()), default=0)
        }