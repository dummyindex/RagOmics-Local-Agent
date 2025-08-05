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
            # If it's a single file (e.g., initial input), handle appropriately
            if input_path.name == "_node_seuratObject.rds":
                shutil.copy2(input_path, execution_dir / "input" / "_node_seuratObject.rds")
            elif input_path.name == "_node_anndata.h5ad":
                # Coming from Python parent
                shutil.copy2(input_path, execution_dir / "input" / "_node_anndata.h5ad")
            else:
                # For initial input or legacy files, copy with original name
                shutil.copy2(input_path, execution_dir / "input" / input_path.name)
                # Also copy as standard name if it's an h5ad or rds file
                if input_path.suffix == ".h5ad":
                    shutil.copy2(input_path, execution_dir / "input" / "_node_anndata.h5ad")
                elif input_path.suffix == ".rds":
                    shutil.copy2(input_path, execution_dir / "input" / "_node_seuratObject.rds")
        
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
    
    # Create path list with only directories
    path_dict <- list(
        input_dir = "/workspace/input",
        output_dir = "/workspace/output"
    )
    
    cat("\\nPath dictionary:\\n")
    print(path_dict)
    cat("\\nParameters:\\n")
    print(params)
    
    # Source function block
    cat("\\nLoading function block\\n")
    source("/workspace/function_block.R")
    
    # Execute function block with path_dict and params
    cat("\\nExecuting function block with path_dict and params\\n")
    run(path_dict, params)  # Pass both path_dict and params
    
    # Verify standard outputs were created (for Seurat/anndata workflows)
    output_r <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    output_py <- file.path(path_dict$output_dir, "_node_anndata.h5ad")
    
    if (file.exists(output_r)) {
        cat(sprintf("\\nStandard R output file _node_seuratObject.rds created successfully\\n"))
    } else if (file.exists(output_py)) {
        cat(sprintf("\\nStandard Python output file _node_anndata.h5ad created\\n"))
    } else {
        # Check for any output files
        all_files <- list.files(path_dict$output_dir, full.names = FALSE)
        if (length(all_files) > 0) {
            cat(sprintf("\\nFound %d output files:\\n", length(all_files)))
            for (f in head(all_files, 5)) {  # Show first 5
                cat(sprintf("  - %s\\n", f))
            }
        } else {
            cat(sprintf("\\nWarning: No output files found in %s\\n", path_dict$output_dir))
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