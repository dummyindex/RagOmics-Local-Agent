"""OpenAI service for LLM interactions."""

import json
from typing import Dict, List, Optional, Any
from openai import OpenAI

from ..utils.logger import get_logger
from ..config import config

logger = get_logger(__name__)


class OpenAIService:
    """Simple service for OpenAI API interactions."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or config.openai_model
        
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """Simple chat completion API call."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            if response_format:
                kwargs["response_format"] = response_format
                
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            logger.debug(f"Raw response content type: {type(content)}")
            logger.debug(f"Raw response content (first 500 chars): {content[:500] if content else 'None'}")
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        json_schema: Dict[str, Any],
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """Chat completion with JSON response format."""
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema
        }
        
        content = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        try:
            logger.debug(f"Attempting to parse JSON content: {content[:500]}...")
            parsed = json.loads(content)
            logger.debug(f"Successfully parsed JSON, type: {type(parsed)}")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {content}")
            raise
    
    def extract_code_block(self, content: str, language: Optional[str] = None) -> Optional[str]:
        """Extract code block from markdown-formatted response."""
        if "```" not in content:
            return content.strip()
            
        parts = content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code blocks are at odd indices
                lines = part.strip().split('\n')
                if lines:
                    # Check if first line is a language identifier
                    first_line = lines[0].lower()
                    if language:
                        # If specific language requested, only return matching blocks
                        if first_line == language:
                            return '\n'.join(lines[1:])
                    else:
                        # Return first code block, removing language identifier if present
                        if first_line in ['python', 'r', 'bash', 'shell']:
                            return '\n'.join(lines[1:])
                        return part.strip()
        
        return None