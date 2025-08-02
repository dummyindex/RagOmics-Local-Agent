"""Enhanced executor for general function block framework."""

import os
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import docker
import pandas as pd

from ..models import (
    AnalysisNode, ExecutionContext, ExecutionRequest, ExecutionResponse,
    NodeExecutionResult, NodeState, FileInfo, FileType, JobInfo, JobStatus,
    FunctionBlockType
)
from ..utils import setup_logger
from .base_executor import BaseExecutor

logger = setup_logger(__name__)


class EnhancedExecutor(BaseExecutor):
    """Enhanced executor that handles general file inputs/outputs."""
    
    def __init__(self, docker_client: docker.DockerClient, 
                 image_name: str, 
                 container_config: Optional[Dict] = None):
        """Initialize enhanced executor."""
        super().__init__(docker_client, image_name, container_config)
        self.file_type_mapping = {
            '.h5ad': FileType.ANNDATA,
            '.csv': FileType.CSV,
            '.tsv': FileType.TSV,
            '.json': FileType.JSON,
            '.parquet': FileType.PARQUET,
            '.h5': FileType.H5,
            '.zarr': FileType.ZARR,
            '.png': FileType.IMAGE,
            '.jpg': FileType.IMAGE,
            '.jpeg': FileType.IMAGE,
            '.pdf': FileType.PDF,
            '.txt': FileType.TEXT
        }
    
    def prepare_execution_environment(self, request: ExecutionRequest, 
                                    workspace_dir: Path) -> Tuple[Path, Path]:
        """Prepare the execution environment with proper file structure."""
        # Create working directory for this execution
        work_dir = workspace_dir / "execution" / request.node.id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        figures_dir = output_dir / "figures"
        
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(exist_ok=True)
        
        # Copy input files
        context = request.execution_context
        for file_info in context.input_files:
            src_path = Path(file_info.filepath)
            if src_path.exists():
                dst_path = input_dir / file_info.filename
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied input file: {file_info.filename}")
        
        # Create execution context JSON
        context_data = {
            "node_id": context.node_id,
            "tree_id": context.tree_id,
            "input_files": [f.dict() for f in context.input_files],
            "available_files": [f.dict() for f in context.available_files],
            "paths": {
                "input_dir": "/workspace/input",
                "output_dir": "/workspace/output",
                "figures_dir": "/workspace/output/figures"
            },
            "tree_metadata": context.tree_metadata,
            "previous_results": context.previous_results
        }
        
        context_file = work_dir / "execution_context.json"
        with open(context_file, 'w') as f:
            json.dump(context_data, f, indent=2, default=str)
        
        return work_dir, output_dir
    
    def create_wrapper_script(self, request: ExecutionRequest, work_dir: Path) -> Path:
        """Create a wrapper script that handles the new framework."""
        node = request.node
        fb = node.function_block
        
        # Determine script extension based on type
        ext = '.py' if fb.type == FunctionBlockType.PYTHON else '.R'
        
        # Create the function block code file
        code_file = work_dir / f"function_block{ext}"
        with open(code_file, 'w') as f:
            f.write(fb.code)
        
        # Create requirements file if needed
        if hasattr(fb, 'requirements') and fb.requirements:
            req_file = work_dir / "requirements.txt"
            with open(req_file, 'w') as f:
                f.write(fb.requirements)
        
        # Create wrapper script
        if fb.type == FunctionBlockType.PYTHON:
            wrapper_content = self._create_python_wrapper(fb.parameters)
        else:
            wrapper_content = self._create_r_wrapper(fb.parameters)
        
        wrapper_file = work_dir / f"run{ext}"
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_content)
        
        return wrapper_file
    
    def _create_python_wrapper(self, parameters: Dict[str, Any]) -> str:
        """Create Python wrapper script."""
        return f'''#!/usr/bin/env python3
"""Enhanced function block wrapper with general file support."""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("function_block_wrapper")

def load_execution_context():
    """Load execution context from JSON file."""
    context_path = Path("/workspace/execution_context.json")
    if context_path.exists():
        with open(context_path, 'r') as f:
            return json.load(f)
    return {{}}

def main():
    """Main execution function."""
    start_time = datetime.now()
    
    # Load execution context
    context = load_execution_context()
    logger.info(f"Loaded execution context for node: {{context.get('node_id', 'unknown')}}")
    
    # Load parameters
    parameters = {json.dumps(parameters)}
    logger.info(f"Loaded parameters: {{parameters}}")
    
    # Create output directories
    output_dir = Path("/workspace/output")
    figures_dir = output_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    try:
        # Import function block
        logger.info("Importing function block")
        sys.path.insert(0, '/workspace')
        from function_block import run
        
        # Execute function block with context
        logger.info("Executing function block")
        result = run(
            context=context,
            parameters=parameters,
            input_dir="/workspace/input",
            output_dir="/workspace/output"
        )
        
        # Handle results
        if result is None:
            result = {{}}
        
        # Save metadata if provided
        if isinstance(result, dict) and 'metadata' in result:
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(result['metadata'], f, indent=2)
            logger.info("Saved metadata")
        
        # Log completion
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Function block completed successfully in {{duration:.2f}} seconds")
        
    except Exception as e:
        logger.error(f"Error during execution: {{str(e)}}")
        logger.error(f"Traceback: {{traceback.format_exc()}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _create_r_wrapper(self, parameters: Dict[str, Any]) -> str:
        """Create R wrapper script."""
        return f'''#!/usr/bin/env Rscript
# Enhanced function block wrapper with general file support

# Load required libraries
suppressPackageStartupMessages({{
    library(jsonlite)
    library(futile.logger)
}})

# Setup logging
flog.appender(appender.file("/workspace/output/execution.log"))
flog.info("Starting R function block execution")

# Load execution context
load_execution_context <- function() {{
    context_path <- "/workspace/execution_context.json"
    if (file.exists(context_path)) {{
        return(fromJSON(context_path))
    }}
    return(list())
}}

# Main execution
main <- function() {{
    start_time <- Sys.time()
    
    # Load context and parameters
    context <- load_execution_context()
    flog.info(paste("Loaded context for node:", context$node_id))
    
    parameters <- fromJSON('{json.dumps(parameters)}')
    flog.info("Loaded parameters")
    
    # Create output directories
    dir.create("/workspace/output", showWarnings = FALSE)
    dir.create("/workspace/output/figures", showWarnings = FALSE)
    
    tryCatch({{
        # Source function block
        source("/workspace/function_block.R")
        
        # Execute function
        result <- run(
            context = context,
            parameters = parameters,
            input_dir = "/workspace/input",
            output_dir = "/workspace/output"
        )
        
        # Save metadata if provided
        if (!is.null(result$metadata)) {{
            write_json(result$metadata, "/workspace/output/metadata.json", pretty = TRUE)
            flog.info("Saved metadata")
        }}
        
        # Log completion
        duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        flog.info(sprintf("Function block completed in %.2f seconds", duration))
        
    }}, error = function(e) {{
        flog.error(paste("Error during execution:", e$message))
        quit(status = 1)
    }})
}}

# Run main function
main()
'''
    
    def execute(self, request: ExecutionRequest, workspace_dir: Path) -> ExecutionResponse:
        """Execute a function block with the enhanced framework."""
        logger.info(f"Executing node {request.node.id} with enhanced framework")
        
        start_time = datetime.now()
        job_id = f"{request.node.id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Prepare execution environment
            work_dir, output_dir = self.prepare_execution_environment(request, workspace_dir)
            
            # Create wrapper script
            wrapper_script = self.create_wrapper_script(request, work_dir)
            
            # Prepare container configuration
            container_config = self.container_config.copy()
            container_config.update({
                'volumes': {
                    str(work_dir): {'bind': '/workspace', 'mode': 'rw'}
                },
                'working_dir': '/workspace',
                'command': self._get_run_command(request.node.function_block.type),
                'environment': {
                    'PYTHONUNBUFFERED': '1',
                    'TZ': 'UTC'
                }
            })
            
            # Set resource limits
            if request.memory_limit:
                container_config['mem_limit'] = request.memory_limit
            if request.cpu_limit:
                container_config['cpu_quota'] = int(request.cpu_limit * 100000)
                container_config['cpu_period'] = 100000
            
            # Run container
            logger.info(f"Starting container for job {job_id}")
            container = self.client.containers.run(
                self.image_name,
                **container_config,
                detach=True
            )
            
            # Wait for completion
            try:
                result = container.wait(timeout=request.timeout_seconds)
                exit_code = result.get('StatusCode', -1)
            except Exception as e:
                logger.error(f"Container timeout or error: {e}")
                container.stop()
                exit_code = -1
            
            # Collect logs
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
            
            # Save logs
            stdout_path = output_dir / "stdout.txt"
            stderr_path = output_dir / "stderr.txt"
            
            with open(stdout_path, 'w') as f:
                f.write(stdout)
            with open(stderr_path, 'w') as f:
                f.write(stderr)
            
            # Clean up container
            container.remove()
            
            # Process results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if exit_code == 0:
                # Collect output files
                output_files = self._collect_output_files(output_dir, request.node.id)
                
                # Load metadata if exists
                metadata = {}
                metadata_file = output_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Create execution result
                execution_result = NodeExecutionResult(
                    node_id=request.node.id,
                    state=NodeState.COMPLETED,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    output_files=output_files,
                    figures=[str(f) for f in (output_dir / "figures").glob("*") if f.is_file()],
                    stdout=stdout,
                    stderr=stderr,
                    metadata=metadata
                )
                
                job_info = JobInfo(
                    job_id=job_id,
                    node_id=request.node.id,
                    container_id=container.short_id,
                    status=JobStatus.SUCCESS,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    exit_code=exit_code,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path)
                )
                
                return ExecutionResponse(
                    success=True,
                    execution_result=execution_result,
                    job_info=job_info
                )
            else:
                # Execution failed
                error_msg = f"Container exited with code {exit_code}"
                if stderr:
                    error_msg += f"\\nStderr: {stderr[-1000:]}"  # Last 1000 chars
                
                return ExecutionResponse(
                    success=False,
                    error=error_msg,
                    job_info=JobInfo(
                        job_id=job_id,
                        node_id=request.node.id,
                        container_id=container.short_id if 'container' in locals() else None,
                        status=JobStatus.FAILED,
                        start_time=start_time,
                        end_time=end_time,
                        duration_seconds=duration,
                        exit_code=exit_code,
                        error_message=error_msg,
                        stdout_path=str(stdout_path),
                        stderr_path=str(stderr_path)
                    )
                )
                
        except Exception as e:
            logger.error(f"Execution error: {str(e)}")
            return ExecutionResponse(
                success=False,
                error=str(e)
            )
    
    def _get_run_command(self, block_type: FunctionBlockType) -> str:
        """Get the run command based on block type."""
        if block_type == FunctionBlockType.PYTHON:
            return "bash -c 'pip install -r requirements.txt 2>/dev/null || true && python run.py'"
        else:
            return "Rscript run.R"
    
    def _collect_output_files(self, output_dir: Path, node_id: str) -> List[FileInfo]:
        """Collect information about output files."""
        output_files = []
        
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path.name not in ["stdout.txt", "stderr.txt", "execution.log"]:
                # Determine file type
                suffix = file_path.suffix.lower()
                file_type = self.file_type_mapping.get(suffix, FileType.OTHER)
                
                # Create file info
                file_info = FileInfo(
                    filename=file_path.name,
                    filepath=str(file_path),
                    filetype=file_type,
                    created_by_node=node_id,
                    created_at=datetime.now()
                )
                
                # Add metadata for specific file types
                if file_type == FileType.CSV:
                    try:
                        df = pd.read_csv(file_path, nrows=5)
                        file_info.metadata = {
                            "shape": list(df.shape),
                            "columns": list(df.columns)
                        }
                    except:
                        pass
                elif file_type == FileType.ANNDATA:
                    file_info.metadata = {"type": "anndata"}
                
                output_files.append(file_info)
        
        return output_files