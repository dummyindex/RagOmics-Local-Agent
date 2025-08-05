#!/usr/bin/env python
"""Debug config loading issue."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set env var before importing
os.environ['OPENAI_API_KEY'] = 'test-key'

print("Attempting to import config...")
try:
    from ragomics_agent_local.config import config
    print(f"Config loaded successfully!")
    print(f"API key: {config.openai_api_key[:10]}..." if config.openai_api_key else "No API key")
    print(f"Model: {config.openai_model}")
except Exception as e:
    print(f"Error loading config: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to create OpenAIService...")
try:
    from ragomics_agent_local.llm_service import OpenAIService
    service = OpenAIService(api_key='test-key')
    print("OpenAIService created successfully!")
except Exception as e:
    print(f"Error creating OpenAIService: {e}")
    import traceback
    traceback.print_exc()