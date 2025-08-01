"""R function block executor."""

from pathlib import Path
from typing import Dict, Any, List
import shutil

from .base_executor import BaseExecutor
from ..models import FunctionBlock, FunctionBlockType, NewFunctionBlock
from ..utils.logger import get_logger
from ..config import config

logger = get_logger(__name__)


class RExecutor(BaseExecutor):
    """Executor for R function blocks."""
    
    @property
    def block_type(self) -> FunctionBlockType:
        return FunctionBlockType.R
    
    @property
    def docker_image(self) -> str:
        return config.r_image
    
    def prepare_execution_dir(
        self, 
        execution_dir: Path,
        function_block: FunctionBlock,
        input_data_path: Path,
        parameters: Dict[str, Any]
    ) -> None:
        """Prepare execution directory for R function block."""
        
        # Create directory structure
        (execution_dir / "input").mkdir(exist_ok=True)
        (execution_dir / "output").mkdir(exist_ok=True)
        (execution_dir / "output" / "figures").mkdir(exist_ok=True)
        
        # Copy input data
        shutil.copy2(input_data_path, execution_dir / "input" / "data.h5ad")
        
        # Write parameters as JSON
        self.write_parameters(execution_dir / "parameters.json", parameters)
        
        # Write function block code
        if isinstance(function_block, NewFunctionBlock):
            # Write the function block code
            with open(execution_dir / "function_block.R", "w") as f:
                f.write(function_block.code)
            
            # Write R package requirements
            with open(execution_dir / "install_packages.R", "w") as f:
                f.write(self._generate_package_install_script(function_block.requirements))
        else:
            raise NotImplementedError("Existing function block execution not yet implemented")
        
        # Write the execution wrapper
        wrapper_code = self._generate_wrapper_code()
        with open(execution_dir / "run.R", "w") as f:
            f.write(wrapper_code)
    
    def get_execution_command(self) -> List[str]:
        """Get R execution command."""
        return [
            "bash", "-c",
            "cd /workspace && " +
            "Rscript install_packages.R && " +
            "Rscript run.R"
        ]
    
    def _generate_package_install_script(self, requirements: str) -> str:
        """Generate R package installation script from requirements."""
        lines = requirements.strip().split('\n')
        install_commands = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if '::' in line:
                # Bioconductor or specific repo package
                if line.startswith('Bioconductor::'):
                    pkg = line.split('::')[1]
                    install_commands.append(f'BiocManager::install("{pkg}")')
                else:
                    install_commands.append(f'remotes::install_github("{line}")')
            else:
                # CRAN package
                install_commands.append(f'install.packages("{line}", repos="https://cloud.r-project.org")')
        
        return f'''#!/usr/bin/env Rscript
# Install required packages

# Ensure BiocManager is installed
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos="https://cloud.r-project.org")

# Ensure remotes is installed for GitHub packages
if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes", repos="https://cloud.r-project.org")

# Install packages
{chr(10).join(install_commands)}

cat("Package installation completed\\n")
'''
    
    def _generate_wrapper_code(self) -> str:
        """Generate wrapper code for R function block execution."""
        return '''#!/usr/bin/env Rscript
# Wrapper script for function block execution

library(anndata)
library(jsonlite)

# Set up logging
log_file <- file("/workspace/output/execution.log", open = "wt")
sink(log_file, type = "output")
sink(log_file, type = "message")

tryCatch({
    cat("Starting function block execution\\n")
    
    # Load parameters
    params <- fromJSON("/workspace/parameters.json")
    cat("Loaded parameters:\\n")
    print(params)
    
    # Load input data
    cat("\\nLoading input data\\n")
    adata <- read_h5ad("/workspace/input/data.h5ad")
    cat(sprintf("Loaded data with dimensions: %d obs x %d vars\\n", 
                nrow(adata$obs), nrow(adata$var)))
    
    # Source function block
    cat("\\nLoading function block\\n")
    source("/workspace/function_block.R")
    
    # Execute function block
    cat("\\nExecuting function block\\n")
    result <- do.call(run, c(list(adata = adata), params))
    
    # Handle results
    if (inherits(result, "AnnData")) {
        # Save the result AnnData
        cat("\\nSaving output data\\n")
        write_h5ad(result, "/workspace/output/output_data.h5ad")
    } else if (is.list(result)) {
        # Function block returned a list with multiple outputs
        if (!is.null(result$adata)) {
            cat("\\nSaving output data from result list\\n")
            write_h5ad(result$adata, "/workspace/output/output_data.h5ad")
        }
        
        # Save any additional metadata
        metadata <- result[names(result) != "adata"]
        if (length(metadata) > 0) {
            write(toJSON(metadata, auto_unbox = TRUE, pretty = TRUE), 
                  "/workspace/output/metadata.json")
        }
    }
    
    cat("\\nFunction block execution completed successfully\\n")
    
}, error = function(e) {
    cat("\\nError during execution:\\n")
    cat(conditionMessage(e), "\\n")
    print(traceback())
    quit(status = 1)
})

# Close log file
sink(type = "output")
sink(type = "message")
close(log_file)
'''