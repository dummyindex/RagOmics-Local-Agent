"""Main agent for orchestrating the hierarchical analysis workflow."""

from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.tree import Tree
from rich.panel import Panel

from .function_selector_agent import FunctionSelectorAgent
from .bug_fixer_agent import BugFixerAgent
from ..models import NodeState, GenerationMode, AnalysisTree, NewFunctionBlock
from ..job_executors import ExecutorManager
from ..llm_service import OpenAIService
from ..analysis_tree_management import AnalysisTreeManager, NodeExecutor
from ..utils.logger import get_logger
from ..utils.data_handler import DataHandler
from ..config import config

logger = get_logger(__name__)
console = Console()


class MainAgent:
    """Main agent that orchestrates the entire analysis workflow."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.executor_manager = ExecutorManager()
        self.llm_service = OpenAIService(api_key=openai_api_key)
        self.tree_manager = AnalysisTreeManager()
        self.node_executor = NodeExecutor(self.executor_manager)
        self.data_handler = DataHandler()
        
        # Initialize sub-agents
        self.function_selector = FunctionSelectorAgent(llm_service=self.llm_service)
        self.bug_fixer = BugFixerAgent(llm_service=self.llm_service)
        
    def run_analysis(
        self,
        input_data_path: str,
        user_request: str,
        output_dir: Optional[str] = None,
        max_nodes: int = 20,
        max_children: int = 3,
        max_debug_trials: int = 3,
        generation_mode: str = "mixed",
        llm_model: str = "gpt-4o-2024-08-06",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run the complete analysis workflow."""
        
        # Convert paths
        input_path = Path(input_data_path)
        output_path = Path(output_dir) if output_dir else config.results_dir / "analysis"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input data not found: {input_path}")
        
        # Load and validate data
        console.print("[bold blue]Loading input data...[/bold blue]")
        adata = self.data_handler.load_data(input_path)
        data_summary = self.data_handler.get_data_summary(adata)
        
        console.print(f"[green]✓[/green] Loaded data: {data_summary['n_obs']} cells × {data_summary['n_vars']} genes")
        
        # Create analysis tree
        mode_map = {
            "mixed": GenerationMode.MIXED,
            "only_new": GenerationMode.ONLY_NEW,
            "only_existing": GenerationMode.ONLY_EXISTING
        }
        
        tree = self.tree_manager.create_tree(
            user_request=user_request,
            input_data_path=str(input_path),
            max_nodes=max_nodes,
            max_children_per_node=max_children,
            max_debug_trials=max_debug_trials,
            generation_mode=mode_map.get(generation_mode, GenerationMode.MIXED),
            llm_model=llm_model
        )
        
        console.print(f"\n[bold]Analysis Request:[/bold] {user_request}")
        console.print(f"[bold]Configuration:[/bold] max_nodes={max_nodes}, max_children={max_children}, mode={generation_mode}\n")
        
        # Main execution loop
        iteration = 0
        satisfied = False
        current_data_path = input_path
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            while not satisfied and self.tree_manager.can_continue_expansion():
                iteration += 1
                
                # Get next batch of nodes to execute
                pending_nodes = self.tree_manager.get_execution_order()
                
                if not pending_nodes:
                    # Generate new nodes
                    task = progress.add_task("Generating analysis plan...", total=1)
                    
                    # Get context for generation
                    if tree.root_node_id:
                        # Find leaf nodes that need expansion
                        leaf_nodes = [
                            node for node in tree.nodes.values()
                            if node.state == NodeState.COMPLETED and not node.children
                        ]
                        
                        for leaf_node in leaf_nodes[:max_children]:  # Limit parallel expansion
                            parent_chain = self.tree_manager.get_parent_chain(leaf_node.id)
                            
                            # Get latest data
                            latest_data_path = self.tree_manager.get_latest_data_path(leaf_node.id)
                            
                            # Use function selector agent
                            context = {
                                'user_request': user_request,
                                'tree': tree,
                                'current_node': leaf_node,
                                'parent_chain': parent_chain,
                                'generation_mode': tree.generation_mode,
                                'max_children': max_children,
                                'data_path': latest_data_path
                            }
                            
                            result = self.function_selector.process(context)
                            
                            satisfied = result['satisfied']
                            
                            if result['function_blocks']:
                                # Add child nodes
                                self.tree_manager.add_child_nodes(leaf_node.id, result['function_blocks'])
                            
                            if satisfied:
                                console.print("[green]✓[/green] Analysis request satisfied!")
                                break
                    else:
                        # Create root node
                        context = {
                            'user_request': user_request,
                            'tree': tree,
                            'current_node': None,
                            'parent_chain': [],
                            'generation_mode': tree.generation_mode,
                            'max_children': 1,  # Root should be single
                            'data_path': input_path
                        }
                        
                        result = self.function_selector.process(context)
                        
                        if result['function_blocks']:
                            self.tree_manager.add_root_node(result['function_blocks'][0])
                    
                    progress.update(task, completed=1)
                    progress.remove_task(task)
                    
                    # Get updated pending nodes
                    pending_nodes = self.tree_manager.get_execution_order()
                
                # Execute pending nodes
                for node in pending_nodes:
                    task = progress.add_task(f"Executing: {node.function_block.name}", total=1)
                    
                    # Update node state
                    self.tree_manager.update_node_execution(node.id, NodeState.RUNNING)
                    
                    # Get input data path
                    node_input_path = self.tree_manager.get_latest_data_path(node.id) or current_data_path
                    
                    # Execute node
                    state, result = self.node_executor.execute_node(
                        node=node,
                        input_data_path=node_input_path,
                        output_base_dir=output_path
                    )
                    
                    # Handle execution result
                    if state == NodeState.COMPLETED:
                        self.tree_manager.update_node_execution(
                            node.id,
                            state=state,
                            output_data_id=result.output_data_path,
                            figures=result.figures,
                            logs=[result.logs] if result.logs else [],
                            duration=result.duration
                        )
                        
                        if result.output_data_path:
                            current_data_path = Path(result.output_data_path)
                            
                        console.print(f"[green]✓[/green] {node.function_block.name}")
                        
                    elif state == NodeState.FAILED:
                        # Try debugging if we haven't exceeded attempts
                        if node.debug_attempts < max_debug_trials:
                            console.print(f"[yellow]![/yellow] {node.function_block.name} failed, attempting to fix...")
                            
                            # Increment debug attempts
                            self.tree_manager.increment_debug_attempts(node.id)
                            
                            # Use bug fixer agent
                            if isinstance(node.function_block, NewFunctionBlock):
                                fix_context = {
                                    'function_block': node.function_block,
                                    'error_message': result.error or "Unknown error",
                                    'stdout': result.stdout,
                                    'stderr': result.stderr,
                                    'analysis_id': tree.id,
                                    'node_id': node.id,
                                    'job_id': result.job_id
                                }
                                
                                fix_result = self.bug_fixer.process(fix_context)
                                
                                if fix_result['success'] and fix_result.get('fixed_code'):
                                    # Update the function block code
                                    node.function_block.code = fix_result['fixed_code']
                                    if fix_result.get('fixed_requirements'):
                                        node.function_block.requirements = fix_result['fixed_requirements']
                                    # Mark as pending to retry
                                    self.tree_manager.update_node_execution(node.id, NodeState.PENDING)
                                    continue
                        
                        # Mark as failed
                        self.tree_manager.update_node_execution(
                            node.id,
                            state=state,
                            error=result.error,
                            logs=[result.logs] if result.logs else [],
                            duration=result.duration
                        )
                        
                        console.print(f"[red]✗[/red] {node.function_block.name}: {result.error}")
                    
                    progress.update(task, completed=1)
                    progress.remove_task(task)
                
                # Save tree state
                tree_path = output_path / "analysis_tree.json"
                self.tree_manager.save_tree(tree_path)
        
        # Display results
        self._display_results(tree, output_path)
        
        # Get summary
        summary = self.tree_manager.get_summary()
        summary["output_dir"] = str(output_path)
        summary["satisfied"] = satisfied
        
        return summary
    
    def _display_results(self, tree: AnalysisTree, output_dir: Path) -> None:
        """Display analysis results."""
        
        # Build tree visualization
        tree_viz = Tree(f"[bold]Analysis Tree[/bold] ({tree.id[:8]}...)")
        
        def add_node_to_viz(node_id: str, parent_branch):
            node = tree.get_node(node_id)
            if not node:
                return
                
            # Node status icon
            status_icon = {
                NodeState.COMPLETED: "[green]✓[/green]",
                NodeState.FAILED: "[red]✗[/red]",
                NodeState.PENDING: "[yellow]○[/yellow]",
                NodeState.RUNNING: "[blue]●[/blue]"
            }.get(node.state, "?")
            
            # Add node to tree
            node_text = f"{status_icon} {node.function_block.name}"
            if node.duration:
                node_text += f" [dim]({node.duration:.1f}s)[/dim]"
                
            branch = parent_branch.add(node_text)
            
            # Add children
            for child_id in node.children:
                add_node_to_viz(child_id, branch)
        
        # Build tree from root
        if tree.root_node_id:
            add_node_to_viz(tree.root_node_id, tree_viz)
        
        console.print("\n")
        console.print(tree_viz)
        
        # Summary panel
        summary = self.tree_manager.get_summary()
        summary_text = f"""
[bold]Analysis Summary[/bold]

Total Nodes: {summary['total_nodes']}
Completed: [green]{summary['completed_nodes']}[/green]
Failed: [red]{summary['failed_nodes']}[/red]
Pending: [yellow]{summary['pending_nodes']}[/yellow]

Total Duration: {summary['total_duration_seconds']:.1f}s
Max Depth: {summary['max_depth']}

Output Directory: {output_dir}
"""
        
        console.print(Panel(summary_text.strip(), title="Results", border_style="green"))
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate the execution environment."""
        return self.executor_manager.validate_environment()