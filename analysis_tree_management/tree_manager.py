"""Analysis tree manager for orchestrating hierarchical analysis."""

import json
import shutil
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
        """Save tree to JSON file and create directory tree markdown."""
        
        if not self.tree:
            logger.warning(f"Cannot save tree to {path} - tree is None")
            return
            
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, "w") as f:
                json.dump(self.tree.model_dump(), f, indent=2, default=str)
            
            logger.info(f"Saved analysis tree to {path}")
            
            # Also create directory tree markdown file
            tree_md_path = path.parent / "directory_tree.md"
            self._create_directory_tree_md(path.parent, tree_md_path)
            
        except Exception as e:
            logger.error(f"Failed to save analysis tree to {path}: {e}")
    
    def _create_directory_tree_md(self, base_dir: Path, output_path: Path) -> None:
        """Create a markdown file with the directory tree structure."""
        try:
            lines = []
            lines.append("# Analysis Tree Directory Structure\n")
            lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append(f"**Location:** `{base_dir}`\n")
            
            # Add node relationship tree as first section
            if self.tree and self.tree.nodes:
                lines.append("\n## Node Execution Tree\n")
                lines.append("This shows the hierarchical execution relationship between nodes:\n")
                lines.append("```\n")
                
                def add_node_tree(node_id: str, prefix: str = "", is_last: bool = True, depth: int = 0):
                    """Add node to the tree visualization."""
                    if node_id not in self.tree.nodes:
                        return
                    
                    node = self.tree.nodes[node_id]
                    connector = "└── " if is_last else "├── "
                    
                    # Format node display with status and ID
                    status_symbol = "✅" if node.state == NodeState.COMPLETED else "❌" if node.state == NodeState.FAILED else "⏳"
                    # Show shortened ID (first 8 chars) for readability
                    short_id = node_id[:8]
                    node_name = f"{node.function_block.name} [{status_symbol}] (node_{short_id}...)"
                    
                    # Add node to tree
                    if depth == 0:
                        lines.append(f"{node_name}\n")
                    else:
                        lines.append(f"{prefix}{connector}{node_name}\n")
                    
                    # Process children
                    children = [nid for nid, n in self.tree.nodes.items() if n.parent_id == node_id]
                    for i, child_id in enumerate(children):
                        is_last_child = (i == len(children) - 1)
                        if depth == 0:
                            child_prefix = ""
                        else:
                            extension = "    " if is_last else "│   "
                            child_prefix = prefix + extension
                        add_node_tree(child_id, child_prefix, is_last_child, depth + 1)
                
                # Find root node(s) and build tree
                root_nodes = [nid for nid, node in self.tree.nodes.items() if node.parent_id is None]
                for i, root_id in enumerate(root_nodes):
                    if i > 0:
                        lines.append("\n")
                    add_node_tree(root_id, "", True, 0)
                
                lines.append("```\n")
            
            # Add directory structure section
            lines.append("\n## Directory Structure\n")
            lines.append("```\n")
            
            # Build the tree structure with better formatting
            def add_item(path: Path, prefix: str = "", is_last: bool = True, depth: int = 0):
                """Recursively add items to the tree."""
                # Skip __pycache__ and .pyc files
                if path.name == "__pycache__" or path.suffix == ".pyc":
                    return
                
                # Skip deep nesting for readability
                if depth > 4 and path.is_dir():
                    # Just show that there are more items
                    connector = "└── " if is_last else "├── "
                    items = list(path.iterdir()) if path.is_dir() else []
                    items = [i for i in items if i.name != "__pycache__" and i.suffix != ".pyc"]
                    if items:
                        lines.append(f"{prefix}{connector}{path.name}/ ... ({len(items)} items)\n")
                    return
                
                # Determine the connector
                connector = "└── " if is_last else "├── "
                
                # Format the item name with appropriate info
                if path.is_file():
                    size = path.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    lines.append(f"{prefix}{connector}{path.name} ({size_str})\n")
                else:
                    # For directories, show cleaner format
                    lines.append(f"{prefix}{connector}{path.name}/\n")
                
                # Process children for directories
                if path.is_dir() and depth < 4:
                    # Get children, excluding hidden files
                    children = sorted([p for p in path.iterdir() 
                                     if not p.name.startswith('.') 
                                     and p.name != "__pycache__"])
                    
                    # Group files and directories
                    dirs = [c for c in children if c.is_dir()]
                    files = [c for c in children if c.is_file()]
                    
                    # Show directories first, then files
                    all_children = dirs + files
                    
                    for i, child in enumerate(all_children):
                        is_last_child = (i == len(all_children) - 1)
                        extension = "    " if is_last else "│   "
                        add_item(child, prefix + extension, is_last_child, depth + 1)
            
            # Start with the base directory
            lines.append(f"{base_dir.name}/\n")
            
            # Process main items with better organization
            main_items = sorted([p for p in base_dir.iterdir() 
                                if not p.name.startswith('.')])
            
            # Group items by type
            json_files = []
            md_files = []
            data_files = []
            main_dirs = []
            tree_dir = None
            
            for item in main_items:
                if item.suffix == '.json':
                    json_files.append(item)
                elif item.suffix == '.md':
                    md_files.append(item)
                elif item.suffix in ['.h5ad', '.csv', '.h5']:
                    data_files.append(item)
                elif item.name.startswith('main_'):
                    main_dirs.append(item)
                elif self.tree and item.name == self.tree.id:
                    tree_dir = item
            
            # Show in organized order
            all_items = json_files + md_files + data_files + main_dirs
            if tree_dir:
                all_items.append(tree_dir)
            
            for i, item in enumerate(all_items):
                is_last = (i == len(all_items) - 1)
                add_item(item, "", is_last, 0)
            
            lines.append("```\n")
            
            # Add summary statistics
            if self.tree:
                lines.append("\n## Analysis Summary\n")
                lines.append(f"- **Tree ID**: {self.tree.id}\n")
                lines.append(f"- **Total Nodes**: {self.tree.total_nodes}\n")
                lines.append(f"- **Completed Nodes**: {self.tree.completed_nodes}\n")
                lines.append(f"- **Failed Nodes**: {self.tree.failed_nodes}\n")
                lines.append(f"- **User Request**: {self.tree.user_request}\n")
                
                # Add node details
                lines.append("\n## Node Details\n")
                for node_id, node in self.tree.nodes.items():
                    status_emoji = "✅" if node.state == NodeState.COMPLETED else "❌" if node.state == NodeState.FAILED else "⏳"
                    lines.append(f"- **{node.function_block.name}** ({status_emoji} {node.state.value})")
                    lines.append(f"  - ID: `{node_id}`")
                    lines.append(f"  - Directory: `nodes/node_{node_id}/`")
                    if node.state == NodeState.FAILED and node.error:
                        lines.append(f"  - Error: {node.error[:100]}...")
            
            # Write to file
            with open(output_path, 'w') as f:
                f.writelines(lines)
            
            logger.info(f"Created directory tree markdown at {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create directory tree markdown: {e}")
    
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
    
    def update_node_state(self, node_id: str, state: NodeState, output_data_id: Optional[str] = None, error: Optional[str] = None):
        """Update the state of a node in the tree.
        
        Args:
            node_id: ID of the node to update
            state: New state for the node
            output_data_id: Output data path if completed
            error: Error message if failed
        """
        if not self.tree or node_id not in self.tree.nodes:
            return
        
        node = self.tree.nodes[node_id]
        old_state = node.state
        node.state = state
        
        # Store error if provided
        if error:
            node.error = error
        
        if output_data_id:
            node.output_data_id = output_data_id
        
        # Update tree counters
        # Convert states to string for comparison if needed
        old_state_str = old_state.value if hasattr(old_state, 'value') else str(old_state)
        state_str = state.value if hasattr(state, 'value') else str(state)
        
        if old_state_str != state_str:
            # Handle completed state changes
            if state == NodeState.COMPLETED or state_str == "completed":
                if old_state != NodeState.COMPLETED and old_state_str != "completed":
                    self.tree.completed_nodes += 1
            elif old_state == NodeState.COMPLETED or old_state_str == "completed":
                if state != NodeState.COMPLETED and state_str != "completed":
                    self.tree.completed_nodes = max(0, self.tree.completed_nodes - 1)
            
            # Handle failed state changes
            if state == NodeState.FAILED or state_str == "failed":
                if old_state != NodeState.FAILED and old_state_str != "failed":
                    self.tree.failed_nodes += 1
            elif old_state == NodeState.FAILED or old_state_str == "failed":
                if state != NodeState.FAILED and state_str != "failed":
                    self.tree.failed_nodes = max(0, self.tree.failed_nodes - 1)
        
        # Tree will be saved by caller if needed
    
    def create_output_structure(self, output_dir: Path) -> Dict[str, Path]:
        """Create the standardized output directory structure.
        
        Args:
            output_dir: Base output directory
            
        Returns:
            Dictionary with paths to key directories
        """
        if not self.tree:
            raise ValueError("No tree initialized")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tree_id_short = self.tree.id[:8]
        
        # Don't create main agent task folder - that's separate from the tree structure
        # The analysis_tree.json should be at the base output directory level
        
        # Create tree folder (just the ID, no "tree_" prefix per docs)
        tree_dir = output_dir / self.tree.id
        tree_dir.mkdir(parents=True, exist_ok=True)
        
        nodes_dir = tree_dir / "nodes"
        nodes_dir.mkdir(exist_ok=True)
        
        # Save analysis tree at base output directory level (per docs)
        self.save_tree(output_dir / "analysis_tree.json")
        
        logger.info(f"Created output structure in {output_dir}")
        
        return {
            "output_dir": output_dir,
            "tree_dir": tree_dir,
            "nodes_dir": nodes_dir,
            "analysis_tree_path": output_dir / "analysis_tree.json"
        }
    
    def create_node_directory(self, node_id: str, nodes_dir: Path) -> Dict[str, Path]:
        """Create directory structure for a single node.
        
        Args:
            node_id: Node ID
            nodes_dir: Base nodes directory
            
        Returns:
            Dictionary with paths to node directories
        """
        if not self.tree or node_id not in self.tree.nodes:
            raise ValueError(f"Node {node_id} not found in tree")
        
        node = self.tree.nodes[node_id]
        node_dir = nodes_dir / f"node_{node_id}"
        node_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        function_block_dir = node_dir / "function_block"
        function_block_dir.mkdir(exist_ok=True)
        
        jobs_dir = node_dir / "jobs"
        jobs_dir.mkdir(exist_ok=True)
        
        outputs_dir = node_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        (outputs_dir / "figures").mkdir(exist_ok=True)
        
        agent_tasks_dir = node_dir / "agent_tasks"
        agent_tasks_dir.mkdir(exist_ok=True)
        
        # Save node info
        node_info = {
            "id": node.id,
            "name": node.function_block.name if node.function_block else "unknown",
            "type": node.function_block.type.value if node.function_block and hasattr(node.function_block.type, 'value') else "python",
            "parent_id": node.parent_id,
            "children_ids": node.children,
            "state": node.state.value if hasattr(node.state, 'value') else str(node.state),
            "level": node.level,
            "created_at": node.created_at.isoformat() if node.created_at else datetime.now().isoformat(),
            "last_execution": node.updated_at.isoformat() if node.updated_at else None,
            "execution_count": 1,
            "debug_attempts": node.debug_attempts
        }
        with open(node_dir / "node_info.json", "w") as f:
            json.dump(node_info, f, indent=2)
        
        # Save function block if present
        if node.function_block:
            self.save_function_block(node.function_block, function_block_dir)
        
        logger.info(f"Created node directory for {node_id}")
        
        return {
            "node_dir": node_dir,
            "function_block_dir": function_block_dir,
            "jobs_dir": jobs_dir,
            "outputs_dir": outputs_dir,
            "agent_tasks_dir": agent_tasks_dir
        }
    
    def save_function_block(self, function_block: Any, function_block_dir: Path) -> None:
        """Save function block to directory.
        
        Args:
            function_block: Function block object
            function_block_dir: Directory to save to
        """
        # Save code
        if hasattr(function_block, 'code'):
            with open(function_block_dir / "code.py", "w") as f:
                f.write(function_block.code)
        
        # Save config
        config = {
            "name": function_block.name,
            "description": function_block.description if hasattr(function_block, 'description') else "",
            "parameters": function_block.parameters if hasattr(function_block, 'parameters') else {},
            "static_config": function_block.static_config.model_dump() if hasattr(function_block, 'static_config') and hasattr(function_block.static_config, 'model_dump') else {}
        }
        with open(function_block_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save requirements
        if hasattr(function_block, 'requirements'):
            with open(function_block_dir / "requirements.txt", "w") as f:
                f.write(function_block.requirements)
    
    def create_job_directory(self, node_id: str, node_dir: Path) -> Path:
        """Create a new job directory for node execution.
        
        Args:
            node_id: Node ID
            node_dir: Node directory
            
        Returns:
            Path to the new job directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{timestamp}_{node_id[:8]}"
        job_dir = node_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories per docs structure
        (job_dir / "input").mkdir(exist_ok=True)  # Added input directory
        (job_dir / "logs").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        (job_dir / "output" / "figures").mkdir(exist_ok=True)
        # NOTE: No past_jobs directory - each job is its own directory under jobs/
        
        # Update latest symlink
        latest_link = node_dir / "jobs" / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(job_id)
        
        logger.info(f"Created job directory {job_id} for node {node_id}")
        
        return job_dir
    
    def save_job_execution_summary(self, job_dir: Path, node_id: str, state: str, 
                                    start_time: datetime, end_time: datetime,
                                    input_path: str, output_path: str,
                                    exit_code: int = 0, error_message: Optional[str] = None) -> None:
        """Save job execution summary.
        
        Args:
            job_dir: Job directory
            node_id: Node ID
            state: Execution state (success/failed)
            start_time: Job start time
            end_time: Job end time
            input_path: Input data path
            output_path: Output data path
            exit_code: Process exit code
            error_message: Error message if failed
        """
        summary = {
            "job_id": job_dir.name,
            "node_id": node_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "exit_code": exit_code,
            "state": state,
            "input_path": input_path,
            "output_path": output_path,
            "error_message": error_message
        }
        
        with open(job_dir / "execution_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    def copy_outputs_to_node(self, job_dir: Path, node_dir: Path) -> None:
        """Copy job outputs to node outputs directory.
        
        Args:
            job_dir: Job directory with outputs
            node_dir: Node directory
        """
        import shutil
        
        job_output_dir = job_dir / "output"
        node_output_dir = node_dir / "outputs"
        
        # Copy main output file
        job_output_file = job_output_dir / "_node_anndata.h5ad"
        if job_output_file.exists():
            shutil.copy2(job_output_file, node_output_dir / "_node_anndata.h5ad")
        
        # Copy figures
        job_figures_dir = job_output_dir / "figures"
        if job_figures_dir.exists():
            node_figures_dir = node_output_dir / "figures"
            if node_figures_dir.exists():
                shutil.rmtree(node_figures_dir)
            shutil.copytree(job_figures_dir, node_figures_dir)
        
        logger.info(f"Copied outputs from {job_dir} to {node_output_dir}")
    
    def get_tree_visualization(self) -> str:
        """Generate a text visualization of the tree.
        
        Returns:
            String representation of the tree structure
        """
        if not self.tree or not self.tree.root_node_id:
            return "Empty tree"
        
        lines = []
        
        def add_node(node_id: str, indent: int = 0):
            node = self.tree.nodes.get(node_id)
            if not node:
                return
            
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            state_str = node.state.value if hasattr(node.state, 'value') else str(node.state)
            lines.append(f"{prefix}{node.function_block.name} ({state_str})")
            
            for child_id in node.children:
                add_node(child_id, indent + 1)
        
        add_node(self.tree.root_node_id)
        return "\n".join(lines)