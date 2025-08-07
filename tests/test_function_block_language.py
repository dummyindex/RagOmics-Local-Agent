#!/usr/bin/env python3
"""Test that function blocks are created with correct language type."""

import pytest
from unittest.mock import Mock

from ragomics_agent_local.agents.function_creator_agent import FunctionCreatorAgent
from ragomics_agent_local.models import FunctionBlockType, NewFunctionBlock, StaticConfig


class TestFunctionBlockLanguage:
    """Test language detection in function block creation."""
    
    def test_extract_function_block_detects_r_type(self):
        """Test that R function blocks are properly detected from LLM response."""
        agent = FunctionCreatorAgent()
        
        # Test R function block with explicit type
        result = {
            "name": "slingshot_analysis",
            "description": "Run Slingshot pseudotime",
            "code": '''run <- function(path_dict, params) {
                library(slingshot)
                # R code
            }''',
            "requirements": "slingshot\nSeurat",
            "type": "r"  # Explicit type
        }
        
        context = {"task_description": "test", "user_request": "test"}
        fb = agent._create_function_block(result, context)
        
        assert isinstance(fb, NewFunctionBlock)
        assert fb.type == FunctionBlockType.R
        assert fb.name == "slingshot_analysis"
        
    def test_extract_function_block_detects_python_type(self):
        """Test that Python function blocks are properly detected from LLM response."""
        agent = FunctionCreatorAgent()
        
        # Test Python function block with explicit type
        result = {
            "name": "palantir_analysis",
            "description": "Run Palantir pseudotime",
            "code": '''def run(path_dict, params):
                import palantir
                # Python code
            ''',
            "requirements": "palantir\nscanpy",
            "type": "python"  # Explicit type
        }
        
        context = {"task_description": "test", "user_request": "test"}
        fb = agent._create_function_block(result, context)
        
        assert isinstance(fb, NewFunctionBlock)
        assert fb.type == FunctionBlockType.PYTHON
        assert fb.name == "palantir_analysis"
        
    def test_language_detection_fallback(self):
        """Test language detection falls back to code analysis when type not specified."""
        agent = FunctionCreatorAgent()
        
        # Test R detection without explicit type
        result_r = {
            "name": "test_r",
            "description": "Test R function",
            "code": '''run <- function(path_dict, params) {
                library(Seurat)
            }''',
            "requirements": "Seurat"
            # No type field
        }
        
        context = {"task_description": "test", "user_request": "test"}
        fb_r = agent._create_function_block(result_r, context)
        assert fb_r.type == FunctionBlockType.R
        
        # Test Python detection without explicit type
        result_py = {
            "name": "test_py",
            "description": "Test Python function",
            "code": '''def run(path_dict, params):
                import scanpy as sc
            ''',
            "requirements": "scanpy"
            # No type field
        }
        
        fb_py = agent._create_function_block(result_py, context)
        assert fb_py.type == FunctionBlockType.PYTHON
        
    def test_language_hints_in_prompt(self):
        """Test that language hints are added to prompts."""
        agent = FunctionCreatorAgent()
        
        # Test R hint
        context_r = {
            "task_description": "Run Slingshot analysis",
            "user_request": "Run Slingshot (R) pseudotime analysis"
        }
        prompt_r = agent._build_creation_prompt(context_r)
        assert "R-specific packages/tools" in prompt_r
        assert "Create an R function block with type='r'" in prompt_r
        
        # Test Python hint
        context_py = {
            "task_description": "Run Palantir analysis",
            "user_request": "Run Palantir (Python) pseudotime analysis"
        }
        prompt_py = agent._build_creation_prompt(context_py)
        assert "Python-specific packages/tools" in prompt_py
        assert "Create a Python function block with type='python'" in prompt_py
        
    def test_common_r_packages_detected(self):
        """Test that common R packages trigger R language detection."""
        agent = FunctionCreatorAgent()
        
        r_packages = ["slingshot", "monocle", "seurat", "deseq"]
        
        for pkg in r_packages:
            context = {
                "task_description": f"Run {pkg} analysis",
                "user_request": f"Use {pkg} for analysis"
            }
            prompt = agent._build_creation_prompt(context)
            assert "R-specific packages/tools" in prompt, f"Failed to detect R for {pkg}"