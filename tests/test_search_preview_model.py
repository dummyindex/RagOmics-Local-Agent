#!/usr/bin/env python3
"""Test that function creator works with gpt-4o-search-preview model."""

import pytest
import os
from unittest.mock import Mock, patch

from ragomics_agent_local.llm_service.openai_service import OpenAIService
from ragomics_agent_local.agents.function_creator_agent import FunctionCreatorAgent
from ragomics_agent_local.models import FunctionBlockType


class TestSearchPreviewModel:
    """Test language detection with gpt-4o-search-preview model."""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="Requires OPENAI_API_KEY environment variable"
    )
    def test_slingshot_with_search_preview_model(self):
        """Test that Slingshot is correctly identified as R with search-preview model."""
        # Initialize LLM service with search-preview model
        llm_service = OpenAIService(model="gpt-4o-search-preview")
        
        # Create function creator agent
        agent = FunctionCreatorAgent(llm_service=llm_service)
        
        # Create context for Slingshot
        context = {
            "task_description": "Run Slingshot (R) pseudotime analysis on single-cell data",
            "user_request": "Run Slingshot (R) – save output to slingshot_pseudotime.csv, load into adata.obs['slingshot']"
        }
        
        # Process the request
        block = agent.process(context)
        
        # Verify result
        assert block is not None, "Function block should be created"
        print(f"\nCreated block: {block.name}")
        print(f"Type: {block.type}")
        print(f"Requirements: {block.requirements}")
        
        # Critical assertion: Should be R type
        assert block.type == FunctionBlockType.R, f"Expected R block but got {block.type}"
        assert "slingshot" in block.name.lower() or "slingshot" in block.code.lower()
        
        # Verify R code structure
        assert "library(" in block.code or "<-" in block.code, "Should have R syntax"
        
        print("✓ Slingshot correctly created as R function block with gpt-4o-search-preview")
        
    def test_search_preview_model_mock(self):
        """Test with mocked search-preview model."""
        # Mock LLM service
        mock_llm = Mock(spec=OpenAIService)
        mock_llm.model = "gpt-4o-search-preview"
        
        # Create agent
        agent = FunctionCreatorAgent(llm_service=mock_llm)
        
        # Mock response
        mock_response = {
            "function_block": {
                "name": "slingshot_analysis",
                "description": "Run Slingshot pseudotime",
                "code": '''run <- function(path_dict, params) {
                    library(slingshot)
                    # R code
                }''',
                "requirements": ["slingshot", "Seurat"],
                "parameters": {},
                "type": "r",
                "static_config": {
                    "args": [],
                    "description": "Run Slingshot pseudotime",
                    "tag": "pseudotime"
                }
            },
            "reasoning": "Slingshot is an R package for pseudotime analysis"
        }
        
        mock_llm.chat_completion_json.return_value = mock_response
        
        # Process request
        context = {
            "task_description": "Run Slingshot analysis",
            "user_request": "Run Slingshot (R) analysis"
        }
        
        block = agent.process(context)
        
        # Verify
        assert block is not None
        assert block.type == FunctionBlockType.R
        assert "slingshot" in block.name.lower()
        
        # Verify the model was used
        assert mock_llm.chat_completion_json.called
        call_args = mock_llm.chat_completion_json.call_args
        print(f"\n✓ Mock test passed with model: {mock_llm.model}")