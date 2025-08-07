#!/usr/bin/env python3
"""Test that function creator correctly identifies language requirements."""

import pytest
from unittest.mock import Mock, patch
import json

from ragomics_agent_local.agents.function_creator_agent import FunctionCreatorAgent
from ragomics_agent_local.llm_service import OpenAIService
from ragomics_agent_local.models import FunctionBlockType, GenerationMode
from ragomics_agent_local.analysis_tree_management.tree_manager import AnalysisTree


class TestLanguageDetection:
    """Test cases for language-specific function block creation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = Mock(spec=OpenAIService)
        self.mock_llm.model = "gpt-4o"  # Add model attribute
        self.agent = FunctionCreatorAgent(llm_service=self.mock_llm)
        
        # Create mock tree for context
        self.mock_tree = Mock(spec=AnalysisTree)
        self.mock_tree.nodes = {}
        
    def _create_context(self, user_request, additional_fields=None):
        """Create a valid context for process_selection_or_creation."""
        context = {
            "user_request": user_request,
            "tree": self.mock_tree,
            "current_node": None,
            "parent_chain": [],
            "generation_mode": GenerationMode.ONLY_NEW,
            "max_children": 1,
            "data_summary": {}
        }
        if additional_fields:
            context.update(additional_fields)
        return context
        
    def test_slingshot_creates_r_block(self):
        """Test that Slingshot pseudotime creates an R function block."""
        # Create context for Slingshot
        context = self._create_context(
            "Run Slingshot (R) – save output to slingshot_pseudotime.csv, load into adata.obs['slingshot']"
        )
        
        # Expected LLM response for process_selection_or_creation
        mock_response = {
            "satisfied": False,
            "reasoning": "Need to run Slingshot pseudotime analysis",
            "next_function_blocks": [{
                "name": "slingshot_pseudotime_analysis",
                "description": "Run Slingshot pseudotime analysis in R",
                "task": "Run Slingshot pseudotime analysis",
                "create_new": True,
                "type": "r",  # Critical: should be 'r' not 'python'
                "requirements": ["Seurat", "slingshot", "Bioconductor::SingleCellExperiment"]
            }]
        }
        
        # Mock for selection/creation response
        self.mock_llm.chat_completion_json.side_effect = [
            mock_response,  # First call for selection
            {  # Second call for creation
                "function_block": {
                    "name": "slingshot_pseudotime_analysis",
                    "description": "Run Slingshot pseudotime analysis in R",
                    "code": '''run <- function(path_dict, params) {
    library(Seurat)
    library(slingshot)
    # R code here
}''',
                    "requirements": ["Seurat", "slingshot", "Bioconductor::SingleCellExperiment"],
                    "parameters": {},
                    "type": "r",
                    "tag": "pseudotime"
                }
            }
        ]
        
        # Process the request
        result = self.agent.process_selection_or_creation(context)
        
        # Verify the result
        assert result["satisfied"] is False  # Should not be satisfied, needs to create new
        assert len(result["function_blocks"]) == 1
        
        # Critical assertion: Check that it's an R block
        block = result["function_blocks"][0]
        assert block.type == FunctionBlockType.R, f"Expected R block but got {block.type}"
        assert block.name == "slingshot_pseudotime_analysis"
        assert "library(slingshot)" in block.code
        assert "Seurat" in block.requirements
        assert "slingshot" in block.requirements
        
    def test_monocle3_creates_r_block(self):
        """Test that Monocle 3 creates an R function block."""
        context = {
            "task_description": "Run Monocle 3 (R) – save output to monocle3_pseudotime.csv",
            "user_request": "Run Monocle 3 (R) – save output to monocle3_pseudotime.csv"
        }
        
        # Expected LLM response for R function block
        mock_response = {
            "function_block": {
                "name": "monocle3_pseudotime_analysis",
                "description": "Run Monocle 3 pseudotime analysis in R",
                "code": '''run <- function(path_dict, params) {
    library(monocle3)
    library(Seurat)
    
    # Code here...
}''',
                "requirements": ["monocle3", "Seurat"],
                "parameters": {},
                "type": "r",
                "static_config": {
                    "args": [],
                    "description": "Run Monocle 3 pseudotime analysis in R",
                    "tag": "pseudotime"
                }
            },
            "reasoning": "Creating R function for Monocle3"
        }
        
        self.mock_llm.chat_completion_json.return_value = mock_response
        block = self.agent.process(context)
        
        assert block is not None
        assert block.type == FunctionBlockType.R
        
    def test_palantir_creates_python_block(self):
        """Test that Palantir (Python) creates a Python function block."""
        context = {
            "task_description": "Run Palantir (Python) – store in adata.obs['palantir']",
            "user_request": "Run Palantir (Python) – store in adata.obs['palantir']"
        }
        
        mock_response = {
            "function_block": {
                "name": "palantir_pseudotime_analysis",
                "description": "Run Palantir pseudotime analysis",
                "code": '''def run(path_dict, params):
    import scanpy as sc
    import palantir
    # Code here...
''',
                "requirements": ["scanpy", "palantir"],
                "parameters": {},
                "type": "python",
                "static_config": {
                    "args": [],
                    "description": "Run Palantir pseudotime analysis",
                    "tag": "pseudotime"
                }
            },
            "reasoning": "Creating Python function for Palantir"
        }
        
        self.mock_llm.chat_completion_json.return_value = mock_response
        block = self.agent.process(context)
        
        assert block is not None
        assert block.type == FunctionBlockType.PYTHON
        
    def test_language_hint_in_request(self):
        """Test that explicit language hints are respected."""
        # Test cases with language hints
        test_cases = [
            ("Run analysis using R package DESeq2", FunctionBlockType.R),
            ("Use Python scanpy for clustering", FunctionBlockType.PYTHON),
            ("Apply Seurat (R) for normalization", FunctionBlockType.R),
            ("Implement using scikit-learn in Python", FunctionBlockType.PYTHON),
        ]
        
        for request, expected_type in test_cases:
            context = {
                "task_description": request,
                "user_request": request
            }
            
            # Mock response with correct type
            mock_response = {
                "function_block": {
                    "name": "test_analysis",
                    "description": "Test analysis",
                    "code": "def run(path_dict, params): pass" if expected_type == FunctionBlockType.PYTHON else "run <- function(path_dict, params) {}",
                    "requirements": ["pandas"] if expected_type == FunctionBlockType.PYTHON else ["Seurat"],
                    "parameters": {},
                    "type": "python" if expected_type == FunctionBlockType.PYTHON else "r",
                    "static_config": {
                        "args": [],
                        "description": "Test analysis",
                        "tag": "analysis"
                    }
                },
                "reasoning": "Creating test function"
            }
            
            self.mock_llm.chat_completion_json.return_value = mock_response
            block = self.agent.process(context)
            
            assert block is not None
            assert block.type == expected_type, f"For request '{request}', expected {expected_type} but got {block.type}"