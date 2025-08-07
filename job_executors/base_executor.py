"""Base class for job executors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import tempfile
import shutil
from datetime import datetime

from ..models import ExecutionResult, FunctionBlock, FunctionBlockType
from ..utils.logger import get_logger
from ..utils.docker_utils import DockerManager
from ..config import config

logger = get_logger(__name__)


class BaseExecutor(ABC):
    """Base class for function block executors."""
    
    def __init__(self, docker_manager: DockerManager):
        self.docker_manager = docker_manager
        self.temp_dir = config.temp_dir
        
    @property
    @abstractmethod
    def block_type(self) -> FunctionBlockType:
        """Return the function block type this executor handles."""
        pass
    
    @property
    @abstractmethod
    def docker_image(self) -> str:
        """Return the Docker image to use."""
        pass
    
    @abstractmethod
    def prepare_execution_dir(
        self, 
        execution_dir: Path,
        function_block: FunctionBlock,
        input_data_path: Path,
        parameters: Dict[str, Any]
    ) -> None:
        """Prepare the execution directory with necessary files."""
        pass
    
    def execute(
        self,
        function_block: FunctionBlock,
        input_data_path: Path,
        output_dir: Path,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a function block in an isolated container."""
        
        start_time = datetime.now()
        execution_id = f"{function_block.id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create temporary execution directory
        execution_dir = self.temp_dir / execution_id
        execution_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare execution directory
            logger.info(f"Preparing execution environment for {function_block.name}")
            self.prepare_execution_dir(
                execution_dir, 
                function_block,
                input_data_path,
                parameters or {}
            )
            
            # Set up volumes
            volumes = {
                str(execution_dir): {
                    "bind": "/workspace",
                    "mode": "rw"
                }
            }
            
            # Set up environment variables
            environment = {
                "COMPUTATION_ID": execution_id,
                "FUNCTION_BLOCK_NAME": function_block.name,
                "INPUT_DATA_PATH": "/workspace/input/_node_anndata.h5ad",
                "OUTPUT_DIR": "/workspace/output",
                "PARAMETERS_PATH": "/workspace/parameters.json",
                "RAGOMICS_LOCAL_MODE": "true"
            }
            
            # Run container
            logger.info(f"Executing function block {function_block.name} in container")
            container_start_time = datetime.now()
            exit_code, stdout, stderr = self.docker_manager.run_container(
                image=self.docker_image,
                command=self.get_execution_command(),
                volumes=volumes,
                environment=environment,
                timeout=config.function_block_timeout
            )
            container_end_time = datetime.now()
            
            # Process results
            duration = (container_end_time - start_time).total_seconds()
            
            # Create result object with all execution details
            if exit_code == 0:
                logger.info(f"Function block {function_block.name} completed successfully")
                result = self.collect_results(execution_dir, output_dir)
                result.duration = duration
                result.logs = stdout
                result.stdout = stdout
                result.stderr = stderr
                result.start_time = container_start_time
                result.end_time = container_end_time
                result.exit_code = exit_code
            else:
                logger.error(f"Function block {function_block.name} failed with exit code {exit_code}")
                # Log stdout/stderr for debugging
                if stdout:
                    logger.debug(f"STDOUT:\n{stdout}")
                if stderr:
                    logger.debug(f"STDERR:\n{stderr}")
                    
                result = ExecutionResult(
                    success=False,
                    error=f"Container exited with code {exit_code}\n{stderr}",
                    logs=stdout + "\n" + stderr,
                    stdout=stdout,
                    stderr=stderr,
                    duration=duration,
                    start_time=container_start_time,
                    end_time=container_end_time,
                    exit_code=exit_code
                )
            
            # Job history is now managed by tree_manager through job directories
            # The deprecated save_job_history method is no longer called
            
            return result
                
        except Exception as e:
            logger.error(f"Error executing function block: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                duration=(datetime.now() - start_time).total_seconds()
            )
        finally:
            # Clean up
            if execution_dir.exists():
                shutil.rmtree(execution_dir)
    
    @abstractmethod
    def get_execution_command(self) -> List[str]:
        """Get the command to execute in the container."""
        pass
    
    def collect_results(self, execution_dir: Path, output_dir: Path) -> ExecutionResult:
        """Collect ALL results from execution directory."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        result = ExecutionResult(success=True)
        
        # Copy ALL files from execution output to job output directory
        execution_output_dir = execution_dir / "output"
        if execution_output_dir.exists():
            # Copy all files in the output directory
            for item in execution_output_dir.glob("*"):
                if item.is_file():
                    dest_path = output_dir / item.name
                    shutil.copy2(item, dest_path)
                    
                    # Track specific file types
                    if item.name == "_node_anndata.h5ad":
                        result.output_data_path = str(dest_path)
                    elif item.name == "_node_seuratObject.rds":
                        # Also track R output files
                        result.output_data_path = str(dest_path)
                    elif item.name == "metadata.json":
                        with open(item) as f:
                            result.metadata = json.load(f)
                elif item.is_dir():
                    # Copy entire subdirectories (e.g., figures/)
                    dest_dir = output_dir / item.name
                    if item.name == "figures":
                        # Special handling for figures directory
                        dest_dir.mkdir(exist_ok=True)
                        for fig_path in item.glob("*"):
                            if fig_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf", ".svg"]:
                                dest_fig = dest_dir / fig_path.name
                                shutil.copy2(fig_path, dest_fig)
                                result.figures.append(str(dest_fig))
                    else:
                        # Copy other directories as-is
                        shutil.copytree(item, dest_dir, dirs_exist_ok=True)
        
        # Collect logs
        log_path = execution_dir / "output" / "execution.log"
        if log_path.exists():
            result.logs += "\n" + log_path.read_text()
        
        return result
    
    def write_parameters(self, path: Path, parameters: Dict[str, Any]) -> None:
        """Write parameters to JSON file."""
        with open(path, "w") as f:
            json.dump(parameters, f, indent=2)
    
    def save_job_history(self, output_dir: Path, result: ExecutionResult, function_block: FunctionBlock) -> None:
        """Save job execution history with stdout, stderr, and metrics.
        
        NOTE: This method is deprecated. Job history should be managed by the
        tree_manager creating separate job directories under nodes/node_id/jobs/.
        This method is kept for backwards compatibility but should not be used
        for new implementations.
        """
        
        # Create past_jobs directory (for backwards compatibility only)
        past_jobs_dir = output_dir / "past_jobs"
        past_jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create job directory with timestamp and status
        job_status = "success" if result.success else "failed"
        job_timestamp = result.start_time.strftime('%Y%m%d_%H%M%S') if result.start_time else datetime.now().strftime('%Y%m%d_%H%M%S')
        job_dir_name = f"{job_timestamp}_{job_status}_{result.job_id[:8]}"
        job_dir = past_jobs_dir / job_dir_name
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save stdout
        if result.stdout:
            stdout_path = job_dir / "stdout.txt"
            stdout_path.write_text(result.stdout)
            logger.debug(f"Saved stdout to {stdout_path}")
        
        # Save stderr
        if result.stderr:
            stderr_path = job_dir / "stderr.txt"
            stderr_path.write_text(result.stderr)
            logger.debug(f"Saved stderr to {stderr_path}")
        
        # Save job metrics as CSV
        import csv
        metrics_path = job_dir / "job_metrics.csv"
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['job_id', result.job_id])
            writer.writerow(['function_block_name', function_block.name])
            writer.writerow(['function_block_type', function_block.type])
            writer.writerow(['start_time', result.start_time.isoformat() if result.start_time else ''])
            writer.writerow(['end_time', result.end_time.isoformat() if result.end_time else ''])
            writer.writerow(['duration_seconds', result.duration])
            writer.writerow(['exit_code', result.exit_code if result.exit_code is not None else ''])
            writer.writerow(['success', result.success])
            writer.writerow(['error', result.error or ''])
            writer.writerow(['output_data_path', result.output_data_path or ''])
            writer.writerow(['num_figures', len(result.figures)])
        logger.debug(f"Saved job metrics to {metrics_path}")
        
        # Save job info as JSON for easier programmatic access
        job_info = {
            "job_id": result.job_id,
            "function_block": {
                "name": function_block.name,
                "type": function_block.type,
                "description": function_block.description,
                "parameters": function_block.parameters
            },
            "execution": {
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_seconds": result.duration,
                "exit_code": result.exit_code,
                "success": result.success,
                "error": result.error
            },
            "outputs": {
                "output_data_path": result.output_data_path,
                "figures": result.figures,
                "metadata": result.metadata
            }
        }
        
        job_info_path = job_dir / "job_info.json"
        with open(job_info_path, 'w') as f:
            json.dump(job_info, f, indent=2)
        logger.debug(f"Saved job info to {job_info_path}")
        
        # NOTE: No current_job symlink - this is handled by tree_manager's job structure
        # The 'latest' symlink in the jobs directory serves this purpose