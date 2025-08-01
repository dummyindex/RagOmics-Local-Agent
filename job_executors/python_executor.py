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
        
        # Copy input data
        shutil.copy2(input_data_path, execution_dir / "input" / "data.h5ad")
        
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
import anndata as ad

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
            parameters = json.load(f)
        logger.info(f"Loaded parameters: {parameters}")
        
        # Load input data
        logger.info("Loading input data")
        adata = ad.read_h5ad('/workspace/input/data.h5ad')
        logger.info(f"Loaded data with shape: {adata.shape}")
        
        # Import and run function block
        logger.info("Importing function block")
        from function_block import run
        
        logger.info("Executing function block")
        result = run(adata, **parameters)
        
        # Handle different return types
        if isinstance(result, ad.AnnData):
            # Save the result AnnData
            logger.info("Saving output data")
            result.write_h5ad('/workspace/output/output_data.h5ad')
        elif isinstance(result, dict):
            # Function block returned a dictionary with multiple outputs
            if 'adata' in result:
                logger.info("Saving output data from result dict")
                result['adata'].write_h5ad('/workspace/output/output_data.h5ad')
            
            # Save any additional metadata
            metadata = {k: v for k, v in result.items() 
                       if k not in ['adata'] and isinstance(v, (str, int, float, list, dict))}
            if metadata:
                with open('/workspace/output/metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        logger.info("Function block execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
'''