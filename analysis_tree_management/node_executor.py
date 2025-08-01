"""Node executor for running individual analysis nodes."""

from pathlib import Path
from typing import Optional, Tuple

from ..models import AnalysisNode, NodeState, ExecutionResult
from ..job_executors import ExecutorManager
from ..utils.logger import get_logger
from ..utils.data_handler import DataHandler
from ..config import config

logger = get_logger(__name__)


class NodeExecutor:
    """Executes individual nodes in the analysis tree."""
    
    def __init__(self, executor_manager: ExecutorManager):
        self.executor_manager = executor_manager
        self.data_handler = DataHandler()
        
    def execute_node(
        self,
        node: AnalysisNode,
        input_data_path: Path,
        output_base_dir: Optional[Path] = None
    ) -> Tuple[NodeState, ExecutionResult]:
        """Execute a single analysis node."""
        
        if output_base_dir is None:
            output_base_dir = config.results_dir
            
        # Create output directory for this node
        output_dir = output_base_dir / node.analysis_id / node.id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Executing node {node.id}: {node.function_block.name}")
            
            # Execute the function block
            result = self.executor_manager.execute(
                function_block=node.function_block,
                input_data_path=input_data_path,
                output_dir=output_dir,
                parameters=node.function_block.parameters
            )
            
            if result.success:
                logger.info(f"Node {node.id} executed successfully")
                return NodeState.COMPLETED, result
            else:
                logger.error(f"Node {node.id} execution failed: {result.error}")
                return NodeState.FAILED, result
                
        except Exception as e:
            logger.error(f"Error executing node {node.id}: {e}")
            return NodeState.FAILED, ExecutionResult(
                success=False,
                error=str(e)
            )
    
    def prepare_node_data(
        self,
        node: AnalysisNode,
        source_data_path: Path
    ) -> Path:
        """Prepare data for node execution."""
        
        # For now, we just use the source data directly
        # In the future, we might need to do transformations
        return source_data_path
    
    def validate_node_results(
        self,
        node: AnalysisNode,
        result: ExecutionResult
    ) -> bool:
        """Validate the results of node execution."""
        
        # Basic validation
        if not result.success:
            return False
            
        # Check if output data exists
        if result.output_data_path:
            output_path = Path(result.output_data_path)
            if not output_path.exists():
                logger.error(f"Output data file not found: {output_path}")
                return False
                
            # Try to load and validate the data
            try:
                adata = self.data_handler.load_data(output_path)
                if adata.n_obs == 0:
                    logger.error("Output data has no observations")
                    return False
            except Exception as e:
                logger.error(f"Failed to load output data: {e}")
                return False
        
        return True