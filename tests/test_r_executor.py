#!/usr/bin/env python3
"""Test R executor functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ragomics_agent_local.job_executors.r_executor import RExecutor
from ragomics_agent_local.models import NewFunctionBlock, FunctionBlockType, StaticConfig


class TestRExecutor:
    """Test cases for R executor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock docker manager
        self.mock_docker_manager = Mock()
        self.executor = RExecutor(docker_manager=self.mock_docker_manager)
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_block_type(self):
        """Test that executor reports correct block type."""
        assert self.executor.block_type == FunctionBlockType.R
        
    def test_docker_image(self):
        """Test that executor uses correct docker image."""
        assert "ragomics-r" in self.executor.docker_image
        
    def test_prepare_execution_dir_creates_structure(self):
        """Test that execution directory structure is created correctly."""
        execution_dir = Path(self.temp_dir) / "execution"
        execution_dir.mkdir()
        
        # Create a simple R function block
        function_block = NewFunctionBlock(
            name="test_r_block",
            description="Test R function block",
            code='''run <- function(path_dict, params) {
                library(Seurat)
                cat("Hello from R\\n")
            }''',
            requirements="Seurat\nggplot2",
            type=FunctionBlockType.R,
            parameters={"test_param": 123},
            static_config=StaticConfig(
                args=[],
                description="Test R function block",
                tag="analysis"
            )
        )
        
        # Create dummy input file
        input_dir = Path(self.temp_dir) / "input"
        input_dir.mkdir()
        input_file = input_dir / "_node_seuratObject.rds"
        input_file.write_text("dummy")
        
        # Prepare execution directory
        self.executor.prepare_execution_dir(
            execution_dir,
            function_block,
            input_dir,
            {"test_param": 456}
        )
        
        # Check directory structure
        assert (execution_dir / "input").exists()
        assert (execution_dir / "output").exists()
        assert (execution_dir / "output" / "figures").exists()
        
        # Check generated files
        assert (execution_dir / "function_block.R").exists()
        assert (execution_dir / "install_packages.R").exists()
        assert (execution_dir / "run.R").exists()
        assert (execution_dir / "parameters.json").exists()
        
        # Check input file was copied
        assert (execution_dir / "input" / "_node_seuratObject.rds").exists()
        
    def test_install_packages_script_generation(self):
        """Test that install_packages.R is generated correctly."""
        execution_dir = Path(self.temp_dir) / "execution"
        execution_dir.mkdir()
        
        # Test various package formats
        function_block = NewFunctionBlock(
            name="test_packages",
            description="Test package installation",
            code="run <- function(path_dict, params) {}",
            requirements="""Seurat
ggplot2
Bioconductor::SingleCellExperiment
Bioconductor::scater
dynverse/princurve""",
            type=FunctionBlockType.R,
            parameters={},
            static_config=StaticConfig(
                args=[],
                description="Test package installation",
                tag="analysis"
            )
        )
        
        # Create dummy input
        input_dir = Path(self.temp_dir) / "input"
        input_dir.mkdir()
        
        self.executor.prepare_execution_dir(
            execution_dir,
            function_block,
            input_dir,
            {}
        )
        
        # Read generated install script
        install_script = (execution_dir / "install_packages.R").read_text()
        
        # Check for proper package installation commands
        assert 'install.packages("Seurat"' in install_script
        assert 'install.packages("ggplot2"' in install_script
        assert 'BiocManager::install("SingleCellExperiment")' in install_script
        assert 'BiocManager::install("scater")' in install_script
        assert 'remotes::install_github("dynverse/princurve")' in install_script
        
        # Check for BiocManager and remotes setup
        assert 'install.packages("BiocManager"' in install_script
        assert 'install.packages("remotes"' in install_script
        
    def test_execution_command(self):
        """Test that execution command runs install_packages.R before main script."""
        command = self.executor.get_execution_command()
        
        # Should be a bash command that runs both scripts
        assert command[0] == "bash"
        assert command[1] == "-c"
        assert "Rscript install_packages.R" in command[2]
        assert "Rscript run.R" in command[2]
        # Install should come before run
        assert command[2].index("install_packages.R") < command[2].index("run.R")
        
    def test_wrapper_code_generation(self):
        """Test that R wrapper code is generated correctly."""
        wrapper_code = self.executor._generate_wrapper_code()
        
        # Check for essential components
        assert "library(jsonlite)" in wrapper_code
        assert "path_dict <- list(" in wrapper_code
        assert "input_dir = " in wrapper_code
        assert "output_dir = " in wrapper_code
        assert "source(\"/workspace/function_block.R\")" in wrapper_code
        assert "run(path_dict, params)" in wrapper_code
        
    def test_empty_requirements_handling(self):
        """Test handling of empty requirements."""
        script = self.executor._generate_package_install_script("")
        
        # Should still have BiocManager and remotes setup
        assert "BiocManager" in script
        assert "remotes" in script
        # But no specific package installations
        assert script.count('install.packages("') == 2  # Only BiocManager and remotes
        
    def test_requirements_with_comments(self):
        """Test that comments in requirements are ignored."""
        requirements = """# Core packages
Seurat
# Visualization
ggplot2
# This is a comment
# Bioconductor::ignored_package
"""
        script = self.executor._generate_package_install_script(requirements)
        
        # Should install Seurat and ggplot2 but not commented package
        assert 'install.packages("Seurat"' in script
        assert 'install.packages("ggplot2"' in script
        assert "ignored_package" not in script