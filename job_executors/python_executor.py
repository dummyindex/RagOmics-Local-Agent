"""Python function block executor."""

from pathlib import Path
from typing import Dict, Any, List
import shutil

from .base_executor import BaseExecutor
from ..models import FunctionBlock, FunctionBlockType, NewFunctionBlock
from ..utils.logger import get_logger
from ..config import config

logger = get_logger(__name__)


class PythonExecutor(BaseExecutor):
    """Executor for Python function blocks."""
    
    @property
    def block_type(self) -> FunctionBlockType:
        return FunctionBlockType.PYTHON
    
    @property
    def docker_image(self) -> str:
        return config.python_image
    
    def prepare_execution_dir(
        self, 
        execution_dir: Path,
        function_block: FunctionBlock,
        input_data_path: Path,
        parameters: Dict[str, Any]
    ) -> None:
        """Prepare execution directory for Python function block."""
        
        # Create directory structure
        (execution_dir / "input").mkdir(exist_ok=True)
        (execution_dir / "output").mkdir(exist_ok=True)
        (execution_dir / "output" / "figures").mkdir(exist_ok=True)
        
        # Copy ALL files from parent's outputs folder to child's input folder
        input_path = Path(input_data_path)
        if input_path.is_dir():
            # If input_data_path is a directory (parent's outputs folder), copy ALL files
            for item in input_path.glob("*"):
                if item.is_file():
                    # Copy with original name to maintain consistency
                    shutil.copy2(item, execution_dir / "input" / item.name)
                elif item.is_dir():
                    # Copy entire subdirectory (e.g., figures/)
                    shutil.copytree(item, execution_dir / "input" / item.name, dirs_exist_ok=True)
        else:
            # If it's a single file (e.g., initial input), copy it with standard name
            if input_path.name == "_node_anndata.h5ad":
                shutil.copy2(input_path, execution_dir / "input" / "_node_anndata.h5ad")
            else:
                # For initial input or legacy files, copy as both original name and standard name
                shutil.copy2(input_path, execution_dir / "input" / input_path.name)
                # Also copy as standard name if it's an h5ad file
                if input_path.suffix == ".h5ad":
                    shutil.copy2(input_path, execution_dir / "input" / "_node_anndata.h5ad")
        
        # Write parameters
        self.write_parameters(execution_dir / "parameters.json", parameters)
        
        # Write function block code
        if isinstance(function_block, NewFunctionBlock):
            # Write the function block code
            with open(execution_dir / "function_block.py", "w") as f:
                f.write(function_block.code)
            
            # Write requirements
            with open(execution_dir / "requirements.txt", "w") as f:
                f.write(function_block.requirements)
        else:
            # For existing function blocks, we would fetch from storage
            # For now, raise an error as we don't have function block storage yet
            raise NotImplementedError("Existing function block execution not yet implemented")
        
        # Write the execution wrapper
        wrapper_code = self._generate_wrapper_code()
        with open(execution_dir / "run.py", "w") as f:
            f.write(wrapper_code)
    
    def get_execution_command(self) -> List[str]:
        """Get Python execution command."""
        return [
            "bash", "-c",
            "cd /workspace && " +
            "pip install -r requirements.txt && " +
            "python run.py"
        ]
    
    def _generate_wrapper_code(self) -> str:
        """Generate wrapper code for function block execution."""
        return '''#!/usr/bin/env python3
"""Wrapper script for function block execution."""

import os
import sys
import json
import logging
import traceback
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/workspace/output/execution.log')
    ]
)
logger = logging.getLogger('function_block_wrapper')

def main():
    """Main execution function."""
    try:
        logger.info("Starting function block execution")
        
        # Load parameters
        with open('/workspace/parameters.json') as f:
            params = json.load(f)
        logger.info(f"Loaded parameters: {params}")
        
        # Create path dictionary with only directories
        path_dict = {
            "input_dir": "/workspace/input",
            "output_dir": "/workspace/output"
        }
        
        logger.info(f"Path dictionary: {path_dict}")
        logger.info(f"Parameters: {params}")
        
        # Import and run function block with path_dict and params
        logger.info("Importing function block")
        from function_block import run
        
        logger.info("Executing function block with path_dict and params")
        run(path_dict, params)  # Pass both path_dict and params
        
        # Verify standard output was created (for anndata workflows)
        standard_output = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
        if os.path.exists(standard_output):
            logger.info(f"Standard output file _node_anndata.h5ad created successfully")
            
            # Try to load and inspect the output to provide context
            try:
                import scanpy as sc
                adata = sc.read_h5ad(standard_output)
                
                # Save data structure information for next nodes
                data_info = {
                    "shape": f"{adata.shape[0]} cells x {adata.shape[1]} genes",
                    "obs_columns": list(adata.obs.columns),
                    "var_columns": list(adata.var.columns),
                    "obsm_keys": list(adata.obsm.keys()),
                    "varm_keys": list(adata.varm.keys()),
                    "uns_keys": list(adata.uns.keys()),
                    "layers": list(adata.layers.keys()) if adata.layers else []
                }
                
                # Save to JSON for next node
                info_file = os.path.join(path_dict["output_dir"], "_data_structure.json")
                with open(info_file, 'w') as f:
                    json.dump(data_info, f, indent=2)
                
                # Also print for logging
                logger.info(f"Output data structure: {json.dumps(data_info, indent=2)}")
                
            except Exception as e:
                logger.warning(f"Could not inspect output data structure: {e}")
        else:
            # Check for any output files
            import glob
            output_files = glob.glob(os.path.join(path_dict["output_dir"], "*"))
            if output_files:
                logger.info(f"Found {len(output_files)} output files")
                for f in output_files[:5]:  # Log first 5
                    logger.info(f"  - {os.path.basename(f)}")
            else:
                logger.warning(f"No output files found in {path_dict['output_dir']}")
        
        logger.info("Function block execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
'''